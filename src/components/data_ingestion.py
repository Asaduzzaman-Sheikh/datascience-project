import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# Create a dataclass for data ingestion configuration
@dataclass
class DataIngestionConfig:
    # Define the file paths for train, test, and raw data
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    # Define a method to initiate data ingestion
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from the specified path
            df = pd.read_csv('notebooks/data/StudentsPerformance.csv')
            logging.info('Read the dataset as pandas dataframe')

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Train test split initiated")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            # Save the training and testing sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Ingestion of the data is completed")

            # Return the paths of the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
