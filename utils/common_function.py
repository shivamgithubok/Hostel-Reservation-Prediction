import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as yaml_file:
            config =  yaml.safe_load(yaml_file)
            logger.info(f"YAML file {file_path} loaded successfully.")
            return config
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise CustomException(f"Error reading YAML file {file_path}: {e}")


def load_data(path):
    try:
        logger.info("Loding data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException(f"Error loading data from {path}: {e}")