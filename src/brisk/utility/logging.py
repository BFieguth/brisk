import logging
import sys 

from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.stderr if record.levelno >= logging.ERROR else sys.stdout
            tqdm.write(msg, file=stream)
            self.flush()

        except Exception:
            self.handleError(record)


class FileFormatter(logging.Formatter):
    def format(self, record):
        spacer_line = "-" * 80
        original_message = super().format(record)
        # Add the spacer before each log entry
        return f"{spacer_line}\n{original_message}\n"
    