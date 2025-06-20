import os
import sys
import json
import logging
import pandas as pd

# Add root to sys.path so we can import modules directly (even if script is deeply nested)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from insurance_premium.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
)
from insurance_premium.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from insurance_premium.components.model_trainer import (
    ModelTrainer,
    ModelTrainerConfig,
)
from insurance_premium.exception import CustomException

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_training_pipeline():
    try:
        logging.info("Starting training pipeline...")

        # Step 1: Ingest data from MySQL and split into train/test CSVs
        ingestion = DataIngestion(DataIngestionConfig(query="SELECT * FROM insurance"))
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Load train.csv for transformation
        df = pd.read_csv(train_path)

        # Step 3: Transform data
        transformer = DataTransformation(DataTransformationConfig())
        X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(df)

        # Step 4: Train model and get best one
        trainer = ModelTrainer(ModelTrainerConfig())
        best_model, metrics = trainer.initiate_model_training(
            X_train, X_test, y_train, y_test
        )

        # Step 5: Save metrics
        metrics_path = os.path.join("artifacts", "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Step 6: Final logging
        best_model_name = metrics.get("Best Model", "Unknown")
        logging.info(f"Training complete. Best model: {best_model_name}")
        logging.info(f"Metrics saved at {metrics_path}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
