from io import StringIO
import logging
import sys
from unittest import mock, TestCase

from brisk.utility.logging import TqdmLoggingHandler, FileFormatter

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
            side_effect=Exception("Mocked formatting exception")
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
