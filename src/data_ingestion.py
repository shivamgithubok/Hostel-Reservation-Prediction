import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self , config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]
        self.test_ratio = self.config["test_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data ingestion started with {self.bucket_name} and file is {self.bucket_file_name}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name) #file name
            blob.download_to_filename(RAW_FILLE_PATH)
            logger.info(f"File {self.bucket_file_name} downloaded from {self.bucket_name} to {RAW_FILLE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading file from GCP: {e}")
            raise CustomException(f"Error downloading file from GCP: {e}")
        
    def split_data(self):
        try:
            df = pd.read_csv(RAW_FILLE_PATH)
            train_df, test_df = train_test_split(df, test_size=self.test_ratio, random_state=42)
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split into train and test sets.")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise CustomException(f"Error splitting data: {e}")
        
    def run(self):
        try:
            logger.info("starting data ingestion")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully.")
        except Exception as ce:
            logger.error(f"Error in data ingestion: {str(ce)}")
        finally:
            logger.info("Data ingestion process finished.")

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()