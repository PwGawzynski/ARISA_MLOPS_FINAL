import typer
from pathlib import Path
import numpy as np
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

from ARISA_DSML.config import (
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash
from mlflow.models.signature import infer_signature

#test commit 1

import os
os.environ["MLFLOW_MODEL_SIGNATURE_INFERENCE_TIMEOUT"] = "60"

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
                    logger.info("Trial finished successfully.")
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
    y_train: pd.DataFrame,
    params: dict | None,
    artifact_name: str = "catboost_model_heart",
    cv_results: pd.DataFrame | None = None,
) -> tuple[str | Path]:
    """Train model on full dataset without cross-validation."""
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}

    with mlflow.start_run():
        loggable_params = params.copy()
        loggable_params["feature_columns"] = list(X_train.columns)
        loggable_params["ignored_features"] = [0]

        catboost_params = {
            k: v for k, v in params.items()
            if k not in ["feature_columns", "ignored_features"]
        }

        model = CatBoostClassifier(
            **catboost_params,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            use_best_model=False,
            plot=True,
        )

        mlflow.log_params(loggable_params)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)
    
        if cv_results is not None:
            cv_metric_mean = cv_results["test-F1-mean"].mean()
            mlflow.log_metric("f1_cv_mean", cv_metric_mean)

            fig1 = plot_error_scatter(
                df_plot=cv_results,
                name="Mean F1 Score",
                title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                xtitle="Training Steps",
                ytitle="Performance Score",
                yaxis_range=[0.5, 1.0],
            )
            mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")

            fig2 = plot_error_scatter(
                cv_results,
                x="iterations",
                y="test-Logloss-mean",
                err="test-Logloss-std",
                name="Mean logloss",
                title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                xtitle="Training Steps",
                ytitle="Logloss",
            )
            mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")
        
        try:
            registered_model_name_to_use = MODEL_NAME if 'MODEL_NAME' in globals() else None
            input_example = X_train.head(5) if isinstance(X_train, pd.DataFrame) else None
            # MLflow model registry
            model_info = mlflow.catboost.log_model(
                cb_model=model,
                artifact_path="mlflow_catboost_model",
                input_example=input_example,
                registered_model_name=registered_model_name_to_use,
            )
        except Exception as e:
            print(f"Error while logging model: {e}")
        
        client = MlflowClient()
        model_info = client.get_latest_versions(MODEL_NAME)[0]
        client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )

        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(loggable_params, model_params_path)

    return (model_path, model_params_path)


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
    logger.info("Loading data")
    # for running in workflow in actions again again
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    y_train = df_train.pop(target)
    X_train = df_train
    print("ELO_1")
    # categorical_indices = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]
    experiment_id = get_or_create_experiment("heart_disease_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, params)
    cv_results = pd.read_csv(cv_output_path)
    print("ELO_2")
    experiment_id = get_or_create_experiment("heart_disease_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(X_train, y_train, params, cv_results=cv_results)

    cv_results = pd.read_csv(cv_output_path)
