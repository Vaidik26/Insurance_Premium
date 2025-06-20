import os
import sys
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

from catboost import CatBoostRegressor
from src.utils import save_object
from src.exception import CustomException

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "R2 Score": r2_score(y_test, y_test_pred),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }

        return metrics, model

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training...")

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    verbose=0,
                    random_state=42,
                ),
            }

            best_model = None
            best_score = float("-inf")
            best_metrics = {}
            best_model_name = ""

            for name, model in models.items():
                logging.info(f"Training {name}...")
                metrics, trained_model = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test
                )
                logging.info(
                    f"{name} R2: {metrics['R2 Score']:.4f}, MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}"
                )

                if metrics["R2 Score"] > best_score:
                    best_score = metrics["R2 Score"]
                    best_model = trained_model
                    best_metrics = metrics
                    best_model_name = name

            # Save the best model
            save_object(self.config.model_path, best_model)
            logging.info(f"Best model: {best_model_name} with R2: {best_score:.4f}")
            logging.info(f"Model saved at: {self.config.model_path}")

            return best_model_name, best_metrics

        except Exception as e:
            raise CustomException(e, sys)
