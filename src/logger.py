import sys
import logging
import os
from datetime import datetime




LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" # This will create a log file name based on the current date and time, formatted as "YYYY-MM-DD_HH-MM-SS.log"
logs_path = os.path.join(os.getcwd(), "logs") # This will create a path for the "logs" directory in the current working directory.
os.makedirs(logs_path, exist_ok = True) # This will create the "logs" directory if it doesn't already exist, and it won't raise an error if the directory already exists.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    level = logging.INFO,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

