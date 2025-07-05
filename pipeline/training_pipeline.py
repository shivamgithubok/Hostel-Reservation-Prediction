from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTrainer
from config.path_config import *
from utils.common_function import read_yaml

if __name__ == "__main__":
    # data ingestion
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()

    #data processing
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH,PROCESSED_DIR, CONFIG_PATH)
    processor.process()
    # model training
    model_trainer = ModelTrainer(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_DIR)
    model_trainer.run()   