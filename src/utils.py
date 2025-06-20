import os
import dill
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException  # Update this path as needed
import sys


def create_directories(paths: list):
    """Create multiple directories if they don't exist."""
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj) -> None:
    """Saves a Python object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Loads a Python object from a dill file."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def get_file_size(path: str) -> str:
    """Returns the file size in KB as a string."""
    try:
        size = os.path.getsize(path) / 1024  # size in KB
        return f"{np.round(size, 2)} KB"
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_regression_model(y_true, y_pred) -> dict:
    """
    Returns regression metrics as a dictionary.
    """
    try:
        metrics = {
            "R2 Score": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        }
        return metrics
    except Exception as e:
        raise CustomException(e, sys)
