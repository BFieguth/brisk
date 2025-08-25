"""Logging related utilities."""
import logging
import logging.handlers
import os
import sys
import pathlib
from typing import Optional

import tqdm

from brisk.services import base

class TqdmLoggingHandler(logging.Handler):
    """A logging handler that writes messages through TQDM.

    This handler ensures that log messages don't interfere with TQDM progress
    bars by using TQDM's write method. Error messages are written to stderr,
    while other messages go to stdout.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write a log record through TQDM.

        Parameters
        ----------
        record : LogRecord
            The log record to be written

        Notes
        -----
        Uses stderr for error messages (level >= ERROR) and stdout for others.
        Preserves TQDM progress bar display by using tqdm.write().
        """
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
    """A custom formatter that adds visual separators between log entries.

    This formatter enhances log readability by adding horizontal lines
    between entries in log files.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with visual separators.

        Parameters
        ----------
        record : LogRecord
            The log record to be formatted

        Returns
        -------
        str
            Formatted log message with separator lines

        Notes
        -----
        Adds an 80-character horizontal line before each log entry.
        """
        spacer_line = "-" * 80
        original_message = super().format(record)
        return f"{spacer_line}\n{original_message}\n"


class LoggingService(base.BaseService):
    """Create logger instance and handle all log messages.
    
    Parameters
    ----------
    name : str
        The name of the service
    results_dir : Path
        The root directory for all results, does not change at runtime
    verbose : bool
        Whether to print verbose output

    Attributes
    ----------
    results_dir : Path
        The root directory for all results, does not change at runtime
    verbose : bool
        Whether to print verbose output
    logger : logging.Logger
        The logger instance
    _memory_handler : Optional[logging.MemoryHandler]
        The memory handler instance
    """
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
        """Configure the logger.
        
        Handles buffering to memory if results_dir is not set, and flushing to
        a file when results_dir becomes available.

        Returns
        -------
        None
        """
        logging.captureWarnings(True)

        logger = logging.getLogger("LoggingService")
        logger.setLevel(logging.DEBUG)

        # Remove all existing handlers to prevent duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = TqdmLoggingHandler()
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
        file_formatter = FileFormatter(
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
        """Set the results directory.

        Parameters
        ----------
        results_dir : Path
            The new results directory
        """
        if self.results_dir == results_dir:
            return
        self.results_dir = results_dir
        self.setup_logger()
