import os
import sys

import numpy as np
import pandas as pd
import pickle   

from src.exception import CustomException
from src.logger import logging

# Define a function to save an object to a file
def save_object(file_path, obj):
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        # Log the successful saving of the object
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

# Define a function to load an object from a file
def load_object(file_path):
    try:
        # Open the file in read-binary mode and load the object using pickle
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

        # Log the successful loading of the object
        logging.info(f"Object loaded successfully from {file_path}")

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)
