import sys
import os 
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

# Create a class for data transformation
@dataclass
class DataTransformationConfig:
    # Define the file path for the preprocessor object
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# Create a class for data transformation functionality
class DataTransformation:
    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    # Define a method to get the data transformer object to perform data transformation on the dataset
    def get_data_transformer_object(self):
        try:
            # Define the numerical and categorical columns in the dataset
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            # Create a pipeline for numerical features
            numerical_pipeline = Pipeline(steps = [
                # Add a step to impute missing values using the median strategy
                ('imputer', SimpleImputer(strategy = 'median')),
                # Add a step to scale the numerical features
                ('scaler', StandardScaler())
            ])

            # Create a pipeline for categorical features
            categorical_pipeline = Pipeline(steps = [
                # Add a step to impute missing values using the most frequent strategy
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                # Add a step to encode categorical features using one-hot encoding
                ('one_hot_encoder', OneHotEncoder())
            ])

            # Logging the successful creation of the numerical and categorical pipelines
            logging.info("Numerical and categorical pipelines have been created successfully")
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            # Combine the numerical and categorical pipelines into a single preprocessor object
            preprocessor = ColumnTransformer(transformers = [
                ('num_pipeline', numerical_pipeline, numerical_features),
                ('cat_pipeline', categorical_pipeline, categorical_features)
            ])

            # Logging the successful creation of the preprocessor object
            logging.info("Data transformation object has been created successfully")

            # Return the preprocessor object
            return preprocessor
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

    # Define a method to initiate data transformation on the training and testing datasets
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training dataset from the specified path
            train_df = pd.read_csv(train_path)
            # Read the testing dataset from the specified path
            test_df = pd.read_csv(test_path)

            # Logging the successful reading of the training and testing datasets
            logging.info("Training and testing datasets have been read successfully")

            # Get the preprocessor object for data transformation
            preprocessing_obj = self.get_data_transformer_object()

            # Seperate the input features and target variable from the training dataset
            input_feature_train_df = train_df.drop(columns = ['math score'])
            target_feature_train_df = train_df['math score']
            

            # Seperate the input features and target variable from the testing dataset
            input_feature_test_df = test_df.drop(columns = ['math score'])
            target_feature_test_df = test_df['math score']

            # Logging the successful separation of input features and target variable
            logging.info("Input features and target variable have been separated successfully") 

            # Apply the preprocessor object to transform the training and testing input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # Logging the successful transformation of the training and testing input features
            logging.info("Training and testing input features have been transformed successfully")      
            # Save the preprocessor object to the specified file path
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            # Pickle the preprocessor object and save it to the specified file path
            logging.info("Preprocessor object has been pickled and saved successfully")

            # Combine the transformed input features and target variable into a single array for training and testing datasets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Logging the successful combination of transformed input features and target variable
            logging.info("Transformed input features and target variable have been combined successfully")

            # Return the transformed training and testing datasets along with the preprocessor object file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example usage of the DataTransformation class
    data_transformation = DataTransformation()

    # Define the paths for the training and testing datasets
    train_path = 'artifacts/train.csv'
    test_path =  'artifacts/test.csv'
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

