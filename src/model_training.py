import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_function import read_yaml, load_data
from scipy.stats import randint, uniform


import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Loding data from {self.train_path}")
            train_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["Booking_Status"])
            y_train = train_df["Booking_Status"]

            X_test = test_df.drop(columns=["Booking_Status"])
            y_test = test_df["Booking_Status"]

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException(f"Error in loading and splitting data: {e}")
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Training LightGBM model")
            model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state=42"])
            logger.info("Starting Randomized Search for hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                n_jobs=self.random_search_params["n_jobs"],
                scoring=self.random_search_params["scoring"]
            )
            logger.info("Starting out Hyperparameter Tuning")
            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best parameters found: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error in training LightGBM model: {e}")
            raise CustomException(f"Error in training LightGBM model: {e}")
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model performance")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise CustomException(f"Error in model evaluation: {e}")
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error in saving model: {e}")
            raise CustomException(f"Error in saving model: {e}")
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")

                logger.info("Starting our MLFLOW experimentation")

                logger.info("Logging to training and testing dataset to mlFLow")
                mlflow.log_artifacts(self.train_path, artifact_path="datasets")
                mlflow.log_artifacts(self.test_path, artifact_path="datasets")
                # Load and split data
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                # Train model
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                # Evaluate model
                evaluation_results = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                # Log model to MLflow
                logger.info("Logging model to MLflow")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")
                logger.info("Logging model parameters and metrics to MLflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics=evaluation_results)
                logger.info("Model training process completed successfully")
                return evaluation_results
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException(f"Error in model training pipeline: {e}")
        

if __name__ == "__main__":
    model_trainer = ModelTrainer(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_DIR)
    model_trainer.run()