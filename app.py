from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
import pandas as pd

# Load your raw DataFrame (from MySQL or CSV)
df = pd.read_csv("artifacts/raw_data.csv")

# Initialize and run
transformer = DataTransformation(DataTransformationConfig())
X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(df)
