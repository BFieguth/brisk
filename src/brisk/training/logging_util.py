"""logging_util.py

This module provides custom logging handlers and formatters to enhance the 
logging capabilities within the Brisk framework. It includes a handler for 
logging progress with TQDM and a custom formatter for file logging that 
adds visual separators between log entries.
"""
import logging
import sys

import tqdm

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = (sys.stderr
                      if record.levelno >= logging.ERROR
                      else sys.stdout)
            tqdm.tqdm.write(msg, file=stream)
            self.flush()

        except (ValueError, TypeError):
            self.handleError(record)


class FileFormatter(logging.Formatter):
    def format(self, record):
        spacer_line = "-" * 80
        original_message = super().format(record)
        # Add the spacer before each log entry
        return f"{spacer_line}\n{original_message}\n"
