"""Logging related utilities."""
import logging
import logging.handlers
import os
import pathlib
from typing import Optional

from brisk.services import base
from brisk.training import logging_util

class LoggingService(base.BaseService):
    """Create logger instance and handle all log messages."""
    def __init__(
        self,
        name: str,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = False
    ):
        super().__init__(name)
        self.results_dir = results_dir
        self.verbose = verbose
        self.logger: logging.Logger = None
        self._memory_handler: Optional[logging.MemoryHandler] = None
        self.setup_logger()

    def setup_logger(self) -> None:
        """
        Configures the logger, handling buffering to memory if results_dir is 
        not set, and flushing to a file when results_dir becomes available.
        """
        logging.captureWarnings(True)

        logger = logging.getLogger("LoggingService")
        logger.setLevel(logging.DEBUG)

        # Remove all existing handlers to prevent duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging_util.TqdmLoggingHandler()
        if self.verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter(
            "\n%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_formatter = logging_util.FileFormatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.results_dir:
            file_handler = logging.FileHandler(
                os.path.join(self.results_dir, "error_log.txt")
            )
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            if self._memory_handler:
                self._memory_handler.setTarget(file_handler)
                self._memory_handler.flush()
        
        elif self._memory_handler:
            self._memory_handler.setTarget(logging.NullHandler())
        else:
            self._memory_handler = logging.handlers.MemoryHandler(
                capacity=1000,
                flushLevel=logging.ERROR,
                target=logging.NullHandler()
            )

        if self._memory_handler:
            logger.addHandler(self._memory_handler)

        self.logger = logger
    
    def set_results_dir(self, results_dir: pathlib.Path) -> None:
        if self.results_dir == results_dir:
            return
        self.results_dir = results_dir
        self.setup_logger()
