import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import logging

from src.utils import save_object, create_directories
from src.exception import CustomException

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    test_size: float = 0.2
    random_state: int = 42


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.target_column = "expenses"

    def get_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            logging.info("Identifying numerical and categorical columns...")

            # Drop target before preprocessing
            features_df = df.drop(columns=[self.target_column])

            numerical_columns = features_df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_columns = features_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df: pd.DataFrame):
        try:
            logging.info("Starting data transformation...")

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            preprocessor = self.get_preprocessor(df)
            preprocessor.fit(X_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the preprocessor
            create_directories([os.path.dirname(self.config.preprocessor_path)])
            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved at {self.config.preprocessor_path}")

            logging.info("Data transformation completed.")
            return (
                X_train_transformed,
                X_test_transformed,
                y_train.to_numpy(),
                y_test.to_numpy(),
            )

        except Exception as e:
            raise CustomException(e, sys)
