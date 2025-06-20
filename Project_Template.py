import os
from pathlib import Path

# ----------------------------------------
# List of all files to be created
# ----------------------------------------
list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitoring.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training.py",
    "src/pipelines/prediction.py",
    "src/logger.py",  # Conditionally written
    "src/exception.py",  # Conditionally written
    "src/utils.py",
    "config/config.yaml",
    "artifacts/",
    "notebooks/eda.ipynb",
    "README.md",
    "requirements.txt",
    "setup.py",  # Conditionally written
    "app.py",
    ".env",
]

# ----------------------------------------
# Create folders and empty files
# ----------------------------------------
for filepath in list_of_files:
    path = Path(filepath)

    # If path has a suffix (like '.py', '.md', etc.), it's a file
    if path.suffix:
        os.makedirs(path.parent, exist_ok=True)
        if not path.exists():
            path.touch()
    else:
        # If there's no suffix, treat it as a directory
        os.makedirs(path, exist_ok=True)


# ----------------------------------------
# Utility to safely write content to file
# ----------------------------------------
def safe_write(file_path: str, content: str):
    """Write to file only if it is empty or doesn't exist."""
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:
            with open(file_path, "w") as f:
                f.write(content)
        else:
            print(f"Skipped writing to {file_path} (already has content)")
    else:
        with open(file_path, "w") as f:
            f.write(content)


# ----------------------------------------
# Write logger.py
# ----------------------------------------
def create_logger_file():
    logger_code = """import logging
import os
from datetime import datetime

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
"""
    safe_write("src/logger.py", logger_code)


# ----------------------------------------
# Write exception.py
# ----------------------------------------
def create_exception_file():
    exception_code = """import sys
from src.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script [{0}] at line [{1}]: {2}".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logging.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
"""
    safe_write("src/exception.py", exception_code)


# ----------------------------------------
# Write setup.py
# ----------------------------------------
def create_setup_file():
    setup_code = """from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Vaidik",
    author_email="vaidiky90@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
"""
    safe_write("setup.py", setup_code)


# ----------------------------------------
# Run only conditional creators
# ----------------------------------------
create_logger_file()
create_exception_file()
create_setup_file()

print("Project template created successfully.")
