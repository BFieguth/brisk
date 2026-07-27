"""Unit tests for the logging adapter.

There is no pre-existing test module for the original ``LoggingService``;
these tests cover the adapter directly. They verify ``LoggerPort`` /
``LoggingServicePort`` conformance (the ports are plain ``Protocol``s, so
conformance is checked structurally) and the buffering / file-handler
lifecycle behaviour the adapter inherits from the original service.
"""

import logging
import uuid

import pytest

from brisk.adapters.logging.logging_adapter import (
    FileFormatter,
    LoggingAdapter,
    TqdmLoggingHandler,
)
from brisk.ports.logger import LoggerPort, LoggingServicePort

# pylint: disable=W0212


LOGGER_PORT_METHODS = ("info", "warning", "error", "exception")
SERVICE_PORT_MEMBERS = (
    "logger", "set_results_dir", "setup_logger", "close_file_handlers"
)


@pytest.fixture()
def adapter():
    """A fresh adapter with a unique logger name (no results dir)."""
    name = f"test_logging_{uuid.uuid4().hex}"
    service = LoggingAdapter(name=name)
    yield service
    service.close_file_handlers()
    for handler in service.logger.handlers[:]:
        handler.close()
        service.logger.removeHandler(handler)


@pytest.mark.unit
class TestLoggingPortConformance:
    def test_logger_satisfies_logger_port(self, adapter):
        for method in LOGGER_PORT_METHODS:
            assert callable(getattr(adapter.logger, method, None))

    def test_adapter_satisfies_logging_service_port(self, adapter):
        for member in SERVICE_PORT_MEMBERS:
            assert hasattr(adapter, member)

    def test_port_protocols_define_expected_members(self):
        # Guard against silent port drift. ``logger`` is a typed attribute
        # (annotation), the rest are methods exposed on the protocol.
        assert set(LOGGER_PORT_METHODS).issubset(set(dir(LoggerPort)))
        service_members = set(dir(LoggingServicePort)) | set(
            LoggingServicePort.__annotations__
        )
        assert set(SERVICE_PORT_MEMBERS).issubset(service_members)


@pytest.mark.unit
class TestLoggingAdapterSetup:
    def test_console_handler_attached(self, adapter):
        assert any(
            isinstance(h, TqdmLoggingHandler) for h in adapter.logger.handlers
        )

    def test_verbose_sets_info_console_level(self):
        name = f"test_logging_{uuid.uuid4().hex}"
        service = LoggingAdapter(name=name, verbose=True)
        try:
            console = next(
                h for h in service.logger.handlers
                if isinstance(h, TqdmLoggingHandler)
            )
            assert console.level == logging.INFO
        finally:
            service.close_file_handlers()

    def test_non_verbose_sets_error_console_level(self, adapter):
        console = next(
            h for h in adapter.logger.handlers
            if isinstance(h, TqdmLoggingHandler)
        )
        assert console.level == logging.ERROR

    def test_no_file_handler_without_results_dir(self, adapter):
        assert not any(
            isinstance(h, logging.FileHandler)
            for h in adapter.logger.handlers
        )

    def test_memory_handler_buffers_without_results_dir(self, adapter):
        assert adapter._memory_handler is not None


@pytest.mark.unit
class TestLoggingAdapterFileLifecycle:
    def test_set_results_dir_creates_file_handler(self, adapter, tmp_path):
        adapter.set_results_dir(tmp_path)
        assert any(
            isinstance(h, logging.FileHandler)
            for h in adapter.logger.handlers
        )
        assert (tmp_path / "error_log.txt").exists()

    def test_warning_written_to_file_after_set_results_dir(
        self, adapter, tmp_path
    ):
        adapter.set_results_dir(tmp_path)
        adapter.logger.warning("warning after results dir")
        adapter.close_file_handlers()  # flush + release the file
        log_path = tmp_path / "error_log.txt"
        assert log_path.exists()
        assert "warning after results dir" in log_path.read_text()

    def test_set_same_results_dir_is_noop(self, adapter, tmp_path):
        adapter.set_results_dir(tmp_path)
        handlers_before = list(adapter.logger.handlers)
        adapter.set_results_dir(tmp_path)
        assert adapter.logger.handlers == handlers_before

    def test_close_file_handlers_removes_them(self, adapter, tmp_path):
        adapter.set_results_dir(tmp_path)
        adapter.close_file_handlers()
        assert not any(
            isinstance(h, logging.FileHandler)
            for h in adapter.logger.handlers
        )


@pytest.mark.unit
class TestFileFormatter:
    def test_adds_separator_line(self):
        formatter = FileFormatter("%(message)s")
        record = logging.LogRecord(
            name="x", level=logging.WARNING, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        out = formatter.format(record)
        assert "-" * 80 in out
        assert "hello" in out
