"""Functions for preprocessing the data."""
from pathlib import Path
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_raw_data(dataset_name: str = DATASET) -> None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)

    # Create the download folder if it doesn't exist
    download_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")

    api.dataset_download_files(dataset_name, path=str(download_folder), unzip=True)

    downloaded_files = list(download_folder.glob("*"))
    if not downloaded_files:
        logger.warning("No files found after download.")
        return None

    latest_file = max(downloaded_files, key=lambda f: f.stat().st_mtime)
    print(f"Latest downloaded file: {latest_file.name}")

    return latest_file.name


def preprocess_df(file: str | Path) -> tuple[Path, Path]:
    df = pd.read_csv(file)

    preprocessor = build_preprocessor()
    transformed_df = preprocess_data(df, preprocessor)

    df_train, df_test = split_data(transformed_df)

    train_path, test_path = save_processed_data(df_train, df_test)

    logger.info(f"Train saved to {train_path}, Test saved to {test_path}")

    return train_path, test_path, df


def build_preprocessor() -> ColumnTransformer:
    smoking_status = [['formerly smoked', 'never smoked', 'smokes', 'Unknown']]
    hot_encoder_occ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder_marital = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot_encoder_gender = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ordinal_encoder_education = OrdinalEncoder()

    return ColumnTransformer(
        transformers=[
        ('occupation', hot_encoder_occ, ['occupation']),
        ('marital_status', hot_encoder_marital, ['marital_status']),
        ('gender', hot_encoder_gender, ['gender']),
        ('education_level', ordinal_encoder_education, ['education_level'])
        ],
        remainder='passthrough'
    )


def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    encoded_data = preprocessor.fit_transform(df)
    return pd.DataFrame(encoded_data, columns=preprocessor.get_feature_names_out())


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=0.2, random_state=42)


def save_processed_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[Path, Path]:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return train_path, test_path


def create_ordinal_encoder(categories_order):
    return OrdinalEncoder(categories=categories_order)


if __name__ == "__main__":
    logger.info("Fetching raw dataset")
    dataset_file_name = get_raw_data()

    # preprocess both sets
    logger.info("preprocessing data")
    preprocess_df(RAW_DATA_DIR / dataset_file_name)