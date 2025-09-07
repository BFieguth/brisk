"""Logging utilities and services for the Brisk package.

This module provides comprehensive logging functionality for the Brisk package,
including custom logging handlers, formatters, and a centralized logging
service. It handles both console and file logging with special support for TQDM
progress bars and memory buffering for delayed file output.

The module includes specialized handlers and formatters to ensure clean log
output that doesn't interfere with progress bars and provides organized
file-based logging with visual separators.

Examples
--------
>>> from brisk.services.logging import LoggingService
>>> from pathlib import Path
>>> 
>>> # Create logging service
>>> logger_service = LoggingService("logging", Path("results"), verbose=True)
>>> 
>>> # Use the logger
>>> logger_service.logger.info("Starting experiment")
>>> logger_service.logger.warning("This is a warning")
>>> logger_service.logger.error("This is an error")
"""

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
    bars by using TQDM's write method. It automatically routes error messages
    to stderr and other messages to stdout, maintaining clean progress bar
    display during long-running operations.

    Notes
    -----
    This handler is essential when using TQDM progress bars in the Brisk
    package, as it prevents log messages from corrupting the progress bar
    display. It uses tqdm.write() which properly handles the terminal
    output formatting.

    Examples
    --------
    >>> import logging
    >>> from brisk.services.logging import TqdmLoggingHandler
    >>> 
    >>> # Create handler and add to logger
    >>> handler = TqdmLoggingHandler()
    >>> logger = logging.getLogger("test")
    >>> logger.addHandler(handler)
    >>> 
    >>> # Log messages won't interfere with TQDM progress bars
    >>> logger.info("This won't break progress bars")
    >>> logger.error("Errors go to stderr")
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
    between entries in log files. It makes it easier to distinguish
    between different log entries when reading log files, especially
    when there are many consecutive log messages.

    Notes
    -----
    The formatter adds an 80-character horizontal line before each
    log entry, making log files more readable and organized.

    Examples
    --------
    >>> import logging
    >>> from brisk.services.logging import FileFormatter
    >>> 
    >>> # Create formatter and add to file handler
    >>> formatter = FileFormatter("%(asctime)s - %(levelname)s - %(message)s")
    >>> file_handler = logging.FileHandler("test.log")
    >>> file_handler.setFormatter(formatter)
    >>> 
    >>> # Log entries will have visual separators
    >>> logger = logging.getLogger("test")
    >>> logger.addHandler(file_handler)
    >>> logger.info("This message will have a separator line above it")
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
    """Centralized logging service for the Brisk package.
    
    This service provides comprehensive logging functionality including console
    and file logging with special support for TQDM progress bars. It handles
    memory buffering for delayed file output and provides organized log
    formatting with visual separators.
    
    The service automatically configures different log levels for console and
    file output, and can buffer log messages in memory when no results directory
    is available, flushing them to file when the directory becomes available.
    
    Attributes
    ----------
    results_dir : Optional[pathlib.Path]
        The root directory for all results and log files
    verbose : bool
        Whether to print verbose output to console
    logger : logging.Logger
        The configured logger instance
    _memory_handler : Optional[logging.MemoryHandler]
        Memory handler for buffering logs when no file output is available
        
    Notes
    -----
    The service uses a memory handler to buffer log messages when no results
    directory is set, allowing logging to work before the directory structure
    is established. When a results directory is provided, logs are written
    to 'error_log.txt' in that directory.
    
    Examples
    --------
    >>> from brisk.services.logging import LoggingService
    >>> from pathlib import Path
    >>> 
    >>> # Create logging service with verbose output
    >>> logger_service = LoggingService("logging", Path("results"))
    >>> 
    >>> # Use the logger
    >>> logger_service.logger.info("Starting experiment")
    >>> logger_service.logger.warning("This is a warning")
    >>> logger_service.logger.error("This is an error")
    >>> 
    >>> # Change results directory
    >>> logger_service.set_results_dir(Path("new_results"))
    """
    def __init__(
        self,
        name: str,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = False
    ) -> None:
        """Initialize the logging service with configuration.
        
        This constructor sets up the logging service with the specified
        configuration and automatically configures the logger with appropriate
        handlers and formatters.
        
        Parameters
        ----------
        name : str
            The name identifier for this service
        results_dir : Optional[pathlib.Path], default=None
            The root directory for log files (if None, logs are buffered in
            memory)
        verbose : bool, default=False
            Whether to enable verbose console output (INFO level vs ERROR level)
            
        Notes
        -----
        If no results_dir is provided, log messages are buffered in memory
        until a results directory is set. This allows logging to work before
        the directory structure is established.
        """
        super().__init__(name)
        self.results_dir = results_dir
        self.verbose = verbose
        self.logger: logging.Logger = None
        self._memory_handler: Optional[logging.MemoryHandler] = None
        self.setup_logger()

    def setup_logger(self) -> None:
        """Configure the logger with appropriate handlers and formatters.
        
        This method sets up the logger with console and file handlers based on
        the current configuration. It handles memory buffering when no results
        directory is available and configures different log levels for console
        and file output.
        
        Notes
        -----
        The method automatically:
        - Removes existing handlers to prevent duplicates
        - Sets up console handler with TQDM support
        - Configures file handler if results_dir is available
        - Sets up memory buffering if no file output is possible
        - Flushes buffered messages when file output becomes available
        """
        logging.captureWarnings(True)

        logger = logging.getLogger("LoggingService")

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
        """Set the results directory and reconfigure logging.

        This method updates the results directory and reconfigures the logger
        to write to the new location. If there were buffered log messages in
        memory, they will be flushed to the new file location.

        Parameters
        ----------
        results_dir : pathlib.Path
            The new results directory for log files

        Notes
        -----
        If the results directory is the same as the current one, no changes
        are made. Otherwise, the logger is reconfigured and any buffered
        messages are flushed to the new file location.

        Examples
        --------
        >>> logger_service = LoggingService("logging", verbose=True)
        >>> # Logs are buffered in memory
        >>> logger_service.logger.info("This will be buffered")
        >>> 
        >>> # Set results directory - buffered logs will be flushed
        >>> logger_service.set_results_dir(Path("results"))
        >>> # Now logs go to results/error_log.txt
        """
        if self.results_dir == results_dir:
            return
        self.results_dir = results_dir
        self.setup_logger()
