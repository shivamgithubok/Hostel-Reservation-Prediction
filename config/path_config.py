import os


####Data ingestion##############

RAW_DIR = "artifact/raw"
RAW_FILLE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


#######Data PProcessing ################

PROCESSED_DIR = "artifact/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed.csv")


#######Model Training ################
MODEL_OUTPUT_DIR = "artifacts/models/lgbm_model.pkl"