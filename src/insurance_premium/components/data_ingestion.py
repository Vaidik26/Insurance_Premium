from dotenv import load_dotenv

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging

from insurance_premium.exception import CustomException
from insurance_premium.utils import create_directories, save_dataframe_as_csv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class DataIngestionConfig:
    query: str
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

        # Load DB credentials from .env
        self.db_config = {
            "host": os.getenv("HOST"),
            "user": os.getenv("USER"),
            "password": os.getenv("PASSWORD"),
            "database": os.getenv("DATABASE"),
        }

        # DEBUG: Print to verify .env values are loaded
        print("DB CONFIG >>>", self.db_config)

    def fetch_data_from_db(self) -> pd.DataFrame:
        try:
            logging.info("Connecting to MySQL database with SQLAlchemy...")

            # Ensure all credentials are present
            if not all(self.db_config.values()):
                raise ValueError("Missing DB credentials in environment variables.")

            # Create SQLAlchemy engine
            engine_str = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
            engine = create_engine(engine_str)

            # Execute query
            df = pd.read_sql(self.config.query, con=engine)

            logging.info("Data fetched successfully from MySQL using SQLAlchemy.")
            return df

        except Exception as e:
            logging.error("Failed to fetch data from MySQL.")
            raise CustomException(e, sys)

    def split_and_save_data(self, df: pd.DataFrame):
        try:
            create_directories(
                [
                    os.path.dirname(self.config.raw_data_path),
                    os.path.dirname(self.config.train_data_path),
                    os.path.dirname(self.config.test_data_path),
                ]
            )

            save_dataframe_as_csv(df, self.config.raw_data_path)
            logging.info(f"Raw data saved at {self.config.raw_data_path}")

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            save_dataframe_as_csv(train_df, self.config.train_data_path)
            save_dataframe_as_csv(test_df, self.config.test_data_path)
            logging.info(f"Train data saved at {self.config.train_data_path}")
            logging.info(f"Test data saved at {self.config.test_data_path}")

        except Exception as e:
            logging.error("Error during data split or saving.")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion...")
        try:
            df = self.fetch_data_from_db()
            self.split_and_save_data(df)
            logging.info("Data ingestion completed.")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
