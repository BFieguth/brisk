"""Logging adapter implementing the logging ports.

Provides a concrete logging implementation behind ``LoggerPort`` and
``LoggingServicePort``. All direct use of the stdlib ``logging`` module and
``tqdm`` lives here so the domain core depends only on the ports.

The adapter mirrors the behaviour of the original ``LoggingService``:
console output routed through TQDM (to avoid corrupting progress bars),
file output with visual separators, and in-memory buffering until a results
directory becomes available.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
import sys

import tqdm


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that writes messages through TQDM.

    Ensures log messages don't interfere with TQDM progress bars by using
    ``tqdm.write``. Error messages are routed to stderr and other messages
    to stdout.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write a log record through TQDM.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be written.
        """
        try:
            msg = self.format(record)
            stream = (
                sys.stderr
                if record.levelno >= logging.ERROR
                else sys.stdout
            )
            tqdm.tqdm.write(msg, file=stream)
            self.flush()
        except (ValueError, TypeError):
            self.handleError(record)


class FileFormatter(logging.Formatter):
    """A formatter that adds visual separators between log entries.

    Prepends an 80-character horizontal line before each log entry to make
    log files easier to read.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with a visual separator.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            Formatted log message with a separator line.
        """
        spacer_line = "-" * 80
        original_message = super().format(record)
        return f"{spacer_line}\n{original_message}\n"


class LoggingAdapter:
    """Logging service adapter implementing ``LoggingServicePort``.

    Configures console and file logging with TQDM support and in-memory
    buffering. The exposed ``logger`` attribute is a standard
    ``logging.Logger`` which structurally satisfies ``LoggerPort``.

    Parameters
    ----------
    name : str, default="LoggingService"
        Logger name used for the underlying ``logging.Logger``.
    results_dir : pathlib.Path, optional
        Directory for the ``error_log.txt`` file. If None, log messages are
        buffered in memory until a results directory is set.
    verbose : bool, default=False
        If True, console output is at INFO level; otherwise ERROR level.

    Examples
    --------
    >>> from pathlib import Path
    >>> service = LoggingAdapter("logging", Path("results"), verbose=True)
    >>> service.logger.info("Starting experiment")
    """

    def __init__(
        self,
        name: str = "LoggingService",
        results_dir: pathlib.Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.results_dir = results_dir
        self.verbose = verbose
        self.logger: logging.Logger = logging.getLogger(name)
        self._memory_handler: logging.handlers.MemoryHandler | None = None
        self.setup_logger()

    def setup_logger(self) -> None:
        """Configure the logger with console, file, and memory handlers.

        Removes any existing handlers to prevent duplicates, attaches a
        TQDM-aware console handler, a file handler when a results directory
        is available, and a memory handler for buffering otherwise.
        """
        logging.captureWarnings(True)

        logger = logging.getLogger(self.name)

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

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
                target=logging.NullHandler(),
            )

        if self._memory_handler:
            logger.addHandler(self._memory_handler)

        self.logger = logger

    def close_file_handlers(self) -> None:
        """Close and remove all file handlers from the logger.

        Releases file locks on log files, which is necessary on Windows
        before the files can be deleted or moved.
        """
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)

    def set_results_dir(self, results_dir: pathlib.Path) -> None:
        """Set the results directory and reconfigure logging.

        Parameters
        ----------
        results_dir : pathlib.Path
            The new results directory for log files. Any buffered messages
            are flushed to the new file location.
        """
        if self.results_dir == results_dir:
            return
        self.results_dir = results_dir
        self.setup_logger()
