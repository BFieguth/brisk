"""Provides custom logging handlers and formatters.

This module enhances the logging capabilities within the Brisk framework by
providing:
    - TqdmLoggingHandler: Handles logging with TQDM progress bars
    - FileFormatter: Adds visual separators between log entries
"""
import logging
import sys

import tqdm

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
