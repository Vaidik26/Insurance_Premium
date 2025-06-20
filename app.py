from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
import pandas as pd
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


# Load your raw DataFrame (from MySQL or CSV)
df = pd.read_csv("artifacts/raw_data.csv")

# Initialize and run
transformer = DataTransformation(DataTransformationConfig())
X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(df)


trainer = ModelTrainer(ModelTrainerConfig())
best_model_name, metrics = trainer.initiate_model_training(
    X_train, X_test, y_train, y_test
)

print(f"\n Best model: {best_model_name}")
print(" Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
