"""Port interfaces for the logging system.

Defines structural contracts for loggers and logging service lifecycle.
"""

import pathlib
from typing import Protocol


class LoggerPort(Protocol):
    """Structured logger with standard log levels."""

    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def exception(self, message: str) -> None: ...


class LoggingServicePort(Protocol):
    """Manages logger lifecycle: creation, file handlers, and teardown."""

    logger: LoggerPort

    def set_results_dir(self, results_dir: pathlib.Path) -> None: ...

    def setup_logger(self) -> None: ...

    def close_file_handlers(self) -> None: ...
