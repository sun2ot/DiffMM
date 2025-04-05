import logging
from datetime import datetime
import os
import sys

# create logger
main_log = logging.getLogger("main")
main_log.setLevel(logging.INFO)

# file handler
log_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
os.makedirs(f"logs", exist_ok=True)
file_handler = logging.FileHandler(f"logs/{log_time}.log")
file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
main_log.addHandler(file_handler)

# stream handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
main_log.addHandler(console_handler)

def olog(message):
    """
    Log a message to both the console and the file handler without duplicate console output.
    """
    # Write to file handler
    for handler in main_log.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(logging.LogRecord(
                name="main", level=logging.INFO, pathname="", lineno=0, msg=message, args=None, exc_info=None
            ))
    
    # Write to console with overwrite effect
    print(f"\r{message}", end="")