import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from brisk.services.logging import LoggingService


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
