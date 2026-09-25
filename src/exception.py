import types
import sys 
import logging 

from src.logger import LOG_FILE_PATH

def error_message_details(error_message, error_details: types.ModuleType):
    _, _, exc_tb = error_details.exc_info()
    
    # Safely check if a traceback exists before accessing attributes
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown File"
        line_number = "Unknown Line"
        
    formatted_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error_message)}]"
    return formatted_message

class CustomException(Exception):
    def __init__(self, message, error_details: types.ModuleType):
        # 1. Generate the fully detailed text first
        self.message = error_message_details(message, error_details=error_details)
        
        # 2. Pass the final detailed text up to the parent class constructor
        super().__init__(self.message)

    # 3. Indented inside the class so Python recognizes it
    def __str__(self):
        return self.message

