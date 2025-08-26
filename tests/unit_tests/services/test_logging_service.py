import logging
import os
from pathlib import Path
from unittest import mock, TestCase
import sys
from io import StringIO

import pytest

from brisk.services.logging import LoggingService, TqdmLoggingHandler, FileFormatter


@pytest.fixture
def logging_service(tmp_path):
    return LoggingService(
        name="logging",
        results_dir=Path(tmp_path),
        verbose=False
    )


@pytest.fixture
def verbose_logging_service(tmp_path):
    return LoggingService(
        name="logging_verbose",
        results_dir=Path(tmp_path),
        verbose=True
    )


@pytest.fixture
def no_results_dir_service():
    return LoggingService(
        name="logging_no_dir",
        results_dir=None,
        verbose=False
    )


class TestTqdmLoggingHandler(TestCase):
    def setUp(self):
        self.handler = TqdmLoggingHandler()
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_emit_to_stdout(self, mock_stdout):
        self.logger.info("Info log message")
        output = mock_stdout.getvalue().strip()
        self.assertIn("Info log message", output)
        self.assertNotIn("ERROR", output)

    @mock.patch("sys.stderr", new_callable=StringIO)
    def test_emit_to_stderr(self, mock_stderr):
        self.logger.error("Error log message")
        output = mock_stderr.getvalue().strip()
        self.assertIn("Error log message", output)

    @mock.patch("tqdm.tqdm.write")
    def test_emit_with_tqdm_write_stdout(self, mock_tqdm_write):
        self.logger.info("Tqdm info log message")
        mock_tqdm_write.assert_called_once_with(
            "Tqdm info log message", file=sys.stdout
            )

    @mock.patch("tqdm.tqdm.write")
    def test_emit_with_tqdm_write_stderr(self, mock_tqdm_write):
        self.logger.error("Tqdm error log message")
        mock_tqdm_write.assert_called_once_with(
            "Tqdm error log message", file=sys.stderr
            )
        
    @mock.patch.object(TqdmLoggingHandler, "handleError")
    def test_emit_exception_handling(self, mock_handle_error):
        with mock.patch.object(
            self.handler, "format", 
            side_effect=ValueError("Mocked formatting exception")
            ):
            self.logger.info("This should trigger an exception in emit")

        mock_handle_error.assert_called_once()


class TestFileFormatter(TestCase):
    def setUp(self):
        self.formatter = FileFormatter()

    def test_format_includes_spacer(self):
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )

        formatted_message = self.formatter.format(record)
        spacer_line = "-" * 80
        self.assertTrue(formatted_message.startswith(spacer_line))
        self.assertIn("Test log message", formatted_message)
        self.assertTrue(formatted_message.endswith("\n"))

    def test_format_output_structure(self):
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname=__file__,
            lineno=20,
            msg="Another test log message",
            args=(),
            exc_info=None,
        )

        formatted_message = self.formatter.format(record)
        spacer_line = "-" * 80
        self.assertTrue(formatted_message.startswith(spacer_line))
        self.assertTrue(formatted_message.endswith("\n"))
        self.assertIn("Another test log message", formatted_message)


class TestLoggingService:
    def test_init_with_results_dir(self, logging_service, tmp_path):
        assert logging_service.name == "logging"
        assert logging_service.results_dir == Path(tmp_path)
        assert logging_service.verbose is False
        assert isinstance(logging_service.logger, logging.Logger)
        assert logging_service.logger.name == "LoggingService"

    def test_init_verbose(self, verbose_logging_service):
        assert verbose_logging_service.verbose is True
        assert any(
            h.level == logging.INFO 
            for h in verbose_logging_service.logger.handlers
        )

    def test_init_without_results_dir(self, no_results_dir_service):
        assert no_results_dir_service.results_dir is None
        assert no_results_dir_service._memory_handler is not None
        assert isinstance(no_results_dir_service._memory_handler, logging.handlers.MemoryHandler)

    def test_logger_setup_file_handler(self, logging_service, tmp_path):
        results_dir = Path(tmp_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        logging_service.setup_logger()
        
        file_handlers = [
            h for h in logging_service.logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        
        expected_log_path = os.path.join(results_dir, "error_log.txt")
        assert any(h.baseFilename == expected_log_path for h in file_handlers)

    def test_set_results_dir_new_path(self, logging_service, tmp_path):
        new_results_dir = Path(tmp_path / "test" / "new_results")
        new_results_dir.mkdir(parents=True, exist_ok=True)
        
        logging_service.set_results_dir(new_results_dir)
        assert logging_service.results_dir == new_results_dir

    @mock.patch('brisk.services.logging.LoggingService.setup_logger')
    def test_set_results_dir_calls_setup_logger(self, mock_setup, logging_service, tmp_path):
        new_results_dir = Path(tmp_path / "test" / "different_results")
        logging_service.set_results_dir(new_results_dir)
        mock_setup.assert_called_once()

    def test_console_handler_levels(self, logging_service):
        # Non-verbose should have ERROR level for console
        console_handlers = [h for h in logging_service.logger.handlers]
        assert any(h.level == logging.ERROR for h in console_handlers)
        
    def test_console_handler_level_verbose(self, verbose_logging_service):
        # Verbose should have INFO level for console
        verbose_console_handlers = [h for h in verbose_logging_service.logger.handlers]
        assert any(h.level == logging.INFO for h in verbose_console_handlers)

    def test_memory_handler_setup_without_results_dir(self, no_results_dir_service):
        assert no_results_dir_service._memory_handler is not None
        assert no_results_dir_service._memory_handler.capacity == 1000
        assert no_results_dir_service._memory_handler.flushLevel == logging.ERROR

    def test_memory_handler_flush_when_results_dir_set(self, tmp_path):
        service = LoggingService("test", results_dir=None, verbose=False)
        assert service._memory_handler is not None
        
        # Set results_dir and verify memory handler gets flushed
        results_dir = Path(tmp_path / "test" / "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with mock.patch.object(service._memory_handler, 'flush') as mock_flush:
            service.set_results_dir(results_dir)
            mock_flush.assert_called_once()
