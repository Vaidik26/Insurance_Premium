from src.components.data_ingestion import DataIngestion, DataIngestionConfig

config = DataIngestionConfig(query="SELECT * FROM insurance")

ingestion = DataIngestion(config)
train_path, test_path = ingestion.initiate_data_ingestion()
print(f"Train data saved at: {train_path}")
print(f"Test data saved at: {test_path}")
