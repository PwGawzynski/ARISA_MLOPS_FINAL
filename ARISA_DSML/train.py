import typer
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
from mlflow.client import MlflowClient
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
import pandas.api.types as ptypes # Import for type checking

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash
import nannyml as nml
import time


app = typer.Typer()


def run_hyperopt(X_train: pd.DataFrame, y_train: np.ndarray, test_size: float = 0.25, n_trials: int = 20, overwrite: bool = False) -> str | Path:
    best_params_path = MODELS_DIR / "best_params.pkl"

    if not best_params_path.is_file() or overwrite:
        logger.info("Running hyperparameter tuning with Optuna.")
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        def objective(trial: optuna.trial.Trial) -> float:
            with mlflow.start_run(nested=True):
                params = {
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
                }
                params.update({
                    "objective": "Logloss",
                    "eval_metric": "Logloss",
                    "random_seed": 42,
                    "allow_empty_features": False,
                })

                model = CatBoostClassifier(**params, verbose=0)
                logger.info(f"Starting trial with params: {params}")
                try:
                    model.fit(
                        X_train_opt,
                        y_train_opt,
                        eval_set=[(X_val_opt, y_val_opt)],
                        early_stopping_rounds=5,
                        verbose=0,
                    )
                    logger.info(f"Trial finished successfully.")
                    preds = model.predict(X_val_opt)
                    probs = model.predict_proba(X_val_opt)

                    f1 = f1_score(y_val_opt, preds)
                    logloss = log_loss(y_val_opt, probs)
                    mlflow.log_metric("val_f1", f1)
                    mlflow.log_metric("val_logloss", logloss)
                    logger.info(f"Trial {trial.number} - Logloss: {logloss:.4f}, F1: {f1:.4f}")

                    return logloss

                except Exception as e:
                    logger.error(f"Error during Optuna trial {trial.number}: {e}")
                    return float('inf')

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study.best_params, best_params_path)
        logger.success(f"Best parameters found and saved to {best_params_path}")
        params = study.best_params
    else:
        logger.info(f"Loading best parameters from {best_params_path}")
        params = joblib.load(best_params_path)

    logger.info("Best Parameters: " + str(params))
    return best_params_path


def train_cv(X_train: pd.DataFrame, y_train: np.ndarray, params: dict, eval_metric: str = "F1", n: int = 5) -> str | Path:
    logger.info(f"Starting cross-validated training (N={n}).")
    cv_params = params.copy()
    cv_params["eval_metric"] = eval_metric
    cv_params["loss_function"] = "Logloss"
    cv_params["random_seed"] = 42
    cv_params["verbose"] = 0

    data = Pool(X_train, y_train)

    logger.info(f"CV parameters: {cv_params}")
    try:
        cv_results = cv(
            params=cv_params,
            pool=data,
            fold_count=n,
            partition_random_seed=42,
            shuffle=True,
            plot=False,
        )
        logger.info("Cross-validation completed.")
    except Exception as e:
        logger.error(f"Error during CV training: {e}")
        raise

    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)
    logger.success(f"CV results saved to {cv_output_path}")

    return cv_output_path


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict | None,
    artifact_name: str = "catboost_model_loan",
    cv_results: pd.DataFrame | None = None,
    label_encoder: LabelEncoder | None = None,
) -> tuple[Path, Path, Path | None]:
    return
    
    if params is None:
        logger.info("Training model without tuned hyperparameters (using CatBoost defaults).")
        params = {}

    full_train_params = params.copy()
    full_train_params.update({
        "objective": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
    })

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id} started for full training.")

        mlflow_logged_params = full_train_params.copy()
        if hasattr(X_train, 'columns'):
            mlflow_logged_params["feature_columns"] = list(X_train.columns)
        if label_encoder is not None:
             mlflow_logged_params["target_classes"] = list(label_encoder.classes_)

        logger.info(f"Logged parameters to MLflow: {mlflow_logged_params}")
        mlflow.log_params(mlflow_logged_params)

        catboost_training_params = {
            k: v for k, v in full_train_params.items()
            if k not in ["feature_columns", "target_classes"]
        }

        model = CatBoostClassifier(
            **catboost_training_params,
            verbose=100,
            early_stopping_rounds=10,
        )
        logger.info(f"Starting CatBoost model training with params: {catboost_training_params}")
        model.fit(
            X_train,
            y_train,
            plot=False,
        )
        logger.info("Model training completed.")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(str(model_file_path))
        mlflow.log_artifact(str(model_file_path), artifact_path="model_cbm_file")
        logger.info(f"Model saved to {model_file_path} and logged as MLflow artifact.")

        le_path = None
        if label_encoder:
             le_path = MODELS_DIR / "label_encoder.pkl"
             joblib.dump(label_encoder, le_path)
             mlflow.log_artifact(str(le_path), artifact_path="label_encoder")
             logger.info(f"LabelEncoder saved to {le_path} and logged.")

        if cv_results is not None and not cv_results.empty:
            logger.info("Logging CV results and figures.")
            if "test-Logloss-mean" in cv_results.columns:
                 cv_logloss_mean = cv_results["test-Logloss-mean"].mean()
                 mlflow.log_metric("logloss_cv_mean", cv_logloss_mean)
                 logger.info(f"Logged CV Logloss mean: {cv_logloss_mean:.4f}")
                 if "test-Logloss-std" in cv_results.columns and "iterations" in cv_results.columns:
                    fig2 = plot_error_scatter(
                        df_plot=cv_results, x="iterations", y="test-Logloss-mean",
                        err="test-Logloss-std",
                        name="Mean logloss",
                        title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                        xtitle="Training Steps", ytitle="Logloss",
                    )
                    if fig2:
                         FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                         fig_path = FIGURES_DIR / "test-logloss-mean_vs_iterations.png"
                         fig2.write_image(str(fig_path))
                         mlflow.log_artifact(str(fig_path))
                         logger.info(f"Logged CV Logloss figure.")
                    else:
                         logger.warning("Failed to create CV Logloss figure.")

            if "test-F1-mean" in cv_results.columns:
                cv_f1_mean_metric = cv_results["test-F1-mean"].mean()
                mlflow.log_metric("f1_cv_mean", cv_f1_mean_metric)
                logger.info(f"Logged CV F1 mean: {cv_f1_mean_metric:.4f}")
                if "test-F1-std" in cv_results.columns and "iterations" in cv_results.columns:
                    fig1 = plot_error_scatter(
                        df_plot=cv_results,
                        name="Mean F1 Score",
                        title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                        xtitle="Training Steps", ytitle="Performance Score",
                    )
                    if fig1:
                         FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                         fig_path = FIGURES_DIR / "test-F1-mean_vs_iterations.png"
                         fig1.write_image(str(fig_path))
                         mlflow.log_artifact(str(fig_path))
                         logger.info(f"Logged CV F1 figure.")
                    else:
                         logger.warning("Failed to create CV F1 figure.")

            mlflow.log_artifact(str(MODELS_DIR / "cv_results.csv"), artifact_path="cv_results")

        else:
            logger.info("No CV results to log.")

        logger.info("Logging model in MLflow format.")
        input_example = X_train.head(5) if isinstance(X_train, pd.DataFrame) else None

        registered_model_name_to_use = MODEL_NAME if 'MODEL_NAME' in globals() else None

        mlflow_model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="mlflow_catboost_model",
            input_example=input_example,
            registered_model_name=registered_model_name_to_use,
        )
        logger.info(f"Model logged via mlflow.catboost.log_model at artifact path: {mlflow_model_info.artifact_path}")

        if registered_model_name_to_use:
            logger.info(f"Attempting to update registry for model: {registered_model_name_to_use}")
            try:
                client = MlflowClient()
                time.sleep(2)

                latest_versions = client.get_latest_versions(registered_model_name_to_use)
                model_version_info = None
                for mv in latest_versions:
                     if mv.run_id == run.info.run_id:
                          model_version_info = mv
                          break

                if model_version_info:
                    client.set_registered_model_alias(registered_model_name_to_use, "challenger", model_version_info.version)
                    logger.info(f"Set alias 'challenger' for {registered_model_name_to_use} version {model_version_info.version}.")

                    git_sha_val = get_git_commit_hash()
                    if git_sha_val:
                        client.set_model_version_tag(
                            name=model_version_info.name, version=model_version_info.version,
                            key="git_sha", value=git_sha_val,
                        )
                        logger.info(f"Set tag 'git_sha': {git_sha_val} for model version {model_version_info.version}.")
                else:
                    logger.warning(f"Could not find the specific model version created by run '{run.info.run_id}' for model '{registered_model_name_to_use}'. Alias/tag not set.")
            except Exception as e:
                 logger.error(f"Error interacting with MLflow registry: {e}")

        else:
            logger.info("Model not registered as no registered_model_name was provided/defined.")

        model_params_file_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(mlflow_logged_params, model_params_file_path)
        logger.info(f"Logged parameters (mlflow_logged_params) saved locally to {model_params_file_path}")
        mlflow.log_artifact(str(model_params_file_path), artifact_path="run_configuration")

        logger.info("Setting up NannyML monitoring objects.")
        reference_df = X_train.copy()

        predictions = model.predict(X_train)
        predicted_probabilities = [p[1] for p in model.predict_proba(X_train)]

        reference_df["prediction"] = predictions
        reference_df["predicted_probability"] = predicted_probabilities
        reference_df[target] = y_train

        try:
            if not ptypes.is_integer_dtype(reference_df["prediction"]):
                 logger.debug(f"Casting 'prediction' column (current dtype: {reference_df['prediction'].dtype}) to int.")
                 reference_df["prediction"] = reference_df["prediction"].astype(int)
            if not ptypes.is_float_dtype(reference_df["predicted_probability"]):
                 logger.debug(f"Casting 'predicted_probability' column (current dtype: {reference_df['predicted_probability'].dtype}) to float.")
                 reference_df["predicted_probability"] = reference_df["predicted_probability"].astype(float)
            if not ptypes.is_integer_dtype(reference_df[target]):
                 logger.debug(f"Casting '{target}' column (current dtype: {reference_df[target].dtype}) to int.")
                 reference_df[target] = reference_df[target].astype(int)

            logger.debug("Successfully ensured NannyML target/prediction/probability columns have integer/float types.")

        except ValueError as e:
            logger.error(f"Failed to cast columns for NannyML: {e}. Check values/dtypes causing error.")
            for col, dtype in [(target, int), ("prediction", int), ("predicted_probability", float)]:
                 try:
                      reference_df[col].astype(dtype)
                 except ValueError:
                       logger.error(f"Casting failed for column '{col}' to {dtype}.")
                       logger.error(f"Value counts of column '{col}':\n{reference_df[col].value_counts().head(20)}")
                       if reference_df[col].dtype == 'object':
                           non_numeric = reference_df[col][pd.to_numeric(reference_df[col], errors='coerce').isna()]
                           if not non_numeric.empty:
                                logger.error(f"Examples of non-numeric values in '{col}': {non_numeric.unique()}")
            raise

        logger.debug("Reference DataFrame dtypes after casting:")
        logger.debug(reference_df[[target, "prediction", "predicted_probability"]].dtypes)
        logger.debug(f"Reference DataFrame head:\n{reference_df[[target, 'prediction', 'predicted_probability']].head()}")
        logger.debug(f"Value counts of reference_df['{target}'] after casting:\n{reference_df[target].value_counts()}")
        logger.debug(f"Value counts of reference_df['prediction'] after casting:\n{reference_df['prediction'].value_counts()}")
        logger.debug(f"Reference_df['predicted_probability'] head after casting:\n{reference_df['predicted_probability'].head()}")

        chunk_size = 50

        feature_column_names = list(X_train.columns)
        udc = nml.UnivariateDriftCalculator(
            column_names=feature_column_names,
            chunk_size=chunk_size,
        )
        udc.fit(reference_df[feature_column_names])
        logger.info("NannyML UnivariateDriftCalculator fitted.")

        estimator = nml.CBPE(
            problem_type="classification_binary",
            y_pred_proba="predicted_probability",
            y_pred="prediction",
            y_true=target,
            metrics=["roc_auc", "f1", "recall", "precision", "accuracy"],
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)
        logger.info("NannyML CBPE estimator fitted.")

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR / "nannyml_store"))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")
        logger.info(f"NannyML objects saved to {store.root_path}")

        mlflow.log_artifact(str(MODELS_DIR / "nannyml_store" / "udc.pkl"), artifact_path="nannyml_objects")
        mlflow.log_artifact(str(MODELS_DIR / "nannyml_store" / "estimator.pkl"), artifact_path="nannyml_objects")
        logger.info("NannyML objects logged to MLflow.")

    logger.info(f"MLflow Run ID: {run.info.run_id} completed.")
    return model_file_path, model_params_file_path, le_path


def plot_error_scatter(
    df_plot: pd.DataFrame,
    x: str = "iterations",
    y: str = "test-F1-mean",
    err: str = "test-F1-std",
    name: str = "",
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    yaxis_range: list[float] | None = None,
) -> go.Figure | None:
    if x not in df_plot.columns or y not in df_plot.columns:
        logger.warning(f"Cannot plot: Missing columns '{x}' or '{y}' in DataFrame.")
        return None
    if err and err not in df_plot.columns:
         logger.warning(f"Cannot plot error band: Missing column '{err}' in DataFrame.")
         err = None

    fig = go.Figure()

    if not name:
        name = y

    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )

    if err:
        x_rev = df_plot[x].iloc[::-1]
        y_upper = df_plot[y] + df_plot[err]
        y_lower = df_plot[y] - df_plot[err]
        y_lower_rev = y_lower.iloc[::-1]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([df_plot[x], x_rev]),
                y=pd.concat([y_upper, y_lower_rev]),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",
                line={"color": "rgba(255, 255, 255, 0)"},
                hoverinfo="skip",
                showlegend=False,
            ),
        )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    return fig


def get_or_create_experiment(experiment_name: str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        logger.info(f"Found existing MLflow experiment: {experiment_name} ({experiment.experiment_id})")
        return experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment: {experiment_name} ({experiment_id})")
        return experiment_id


if __name__ == "__main__":
    logger.info("Starting training pipeline.")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        logger.info(f"Loaded training data from {PROCESSED_DATA_DIR / 'train.csv'}. Shape: {df_train.shape}")

        if target in df_train.columns:
            logger.debug(f"Original '{target}' column dtype: {df_train[target].dtype}")
            logger.debug(f"Original '{target}' column value counts:\n{df_train[target].value_counts()}")
        else:
            logger.error(f"Target column '{target}' not found in the loaded training data.")
            raise ValueError(f"Target column '{target}' not found.")

        y_train_original = df_train.pop(target)
        X_train = df_train
        logger.info(f"Separated target '{target}'. X_train shape: {X_train.shape}, y_train shape: {y_train_original.shape}")

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_original)
        logger.info(f"Encoded target variable using LabelEncoder. Classes: {le.classes_}")
        logger.debug(f"Encoded y_train dtype: {y_train_encoded.dtype}")
        logger.debug(f"Encoded y_train first 5 values: {y_train_encoded[:5]}")

        experiment_id_hp = get_or_create_experiment("loan_prediction_hyperparam_tuning")
        mlflow.set_experiment(experiment_id=experiment_id_hp)
        logger.info("Running hyperparameter tuning.")
        best_params_path = run_hyperopt(X_train, y_train_encoded)
        params = joblib.load(best_params_path)
        logger.success("Hyperparameter tuning completed.")

        experiment_id_cv = get_or_create_experiment("loan_prediction_cv_training")
        mlflow.set_experiment(experiment_id=experiment_id_cv)
        logger.info("Running cross-validation.")
        cv_output_path = train_cv(X_train, y_train_encoded, params)
        cv_results = pd.read_csv(cv_output_path)
        logger.success("Cross-validation completed.")

        experiment_id_full = get_or_create_experiment("loan_prediction_full_training")
        mlflow.set_experiment(experiment_id=experiment_id_full)
        logger.info("Starting final model training on full data.")
        model_path, model_params_path, le_path = train(
            X_train,
            y_train_encoded,
            params,
            cv_results=cv_results,
            label_encoder=le
        )
        logger.success("Full model training completed.")

        print("\n--- Training Pipeline Completed ---")
        print(f"Best parameters saved at: {best_params_path}")
        print(f"CV results saved at: {cv_output_path}")
        print(f"Final model saved at: {model_path}")
        print(f"Final model parameters saved at: {model_params_path}")
        if le_path:
            print(f"Label Encoder saved at: {le_path}")
        print(f"MLflow Run ID (Full Train): {mlflow.active_run().info.run_id}")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Ensure processed data exists.")
    except ValueError as e:
         logger.error(f"Data processing error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the training pipeline: {e}")