import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from  src.custom_exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path, test_path,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")

            logger.info("Dropping the columns")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying labelEncoding categorical columns")
            labe_encoder = LabelEncoder()

            mapping = {}
            for col in cat_cols:
                df[col] = labe_encoder.fit_transform(df[col])
                mapping[col] = {label:code for label,code in zip(labe_encoder.classes_, labe_encoder.transform(labe_encoder.classes_))}
            logger.info("Label MApping are: ")
            for col,mapping in mapping.items():
                logger.info(f"{col}: {mapping}")

            logger.info("doing skewness throshold")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise CustomException(f"Error in data preprocessing: {e}")
        
    def balance_data(self,df):
        try:
            logger.info("Starting data balancing")
            X = df.drop(columns=["Booking_Status"])
            y = df["Booking_Status"]

            logger.info("Applying SMOTE for oversampling")
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["Booking_Status"] = y_resampled

            logger.info("Data balancing completed")
            return balanced_df
        except Exception as e:
            logger.error(f"Error in data balancing: {e}")
            raise CustomException(f"Error in data balancing: {e}")
        
    def select_features(self,df):
        try:
            logger.info("Starting feature selection")
            X = df.drop(columns=["Booking_Status"])
            y = df["Booking_Status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": feature_importances
                })
            top_features_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)

            num_features_to_select = self.config["data_processing"]["num_features_to_select"]
            top_10_features = top_features_importance_df["feature"].head(10).values

            top_10_df = df[top_10_features.tolist() + ["Booking_Status"]]

            logger.info("Feature selection completed")
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException(f"Error in feature selection: {e}")
        
    def save_data(self,df,file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException(f"Error saving data to {file_path}: {e}")

    def process(self):
        try:
            logger.info("Loading data freom raw directory")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error("Error during data processing")
            raise CustomException(f"Error during data processing: {e}")
        

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH,PROCESSED_DIR, CONFIG_PATH)
    processor.process()