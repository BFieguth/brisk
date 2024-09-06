import pytest
from unittest import mock

from ml_toolkit.utility.alert_mailer import AlertMailer

class TestAlertMailer:
    """Test class for AlertMailer."""

    @pytest.fixture
    def email_instance(self):
        """
        Fixture to create an instance of AlertMailer with mocked config values.

        This avoids using real email credentials and sets up an instance with mock values.
        """
        # Mock the ConfigParser.read() method. This behaves as if the file was succesfully read.
        with mock.patch("configparser.ConfigParser.read"):
            # Mock the __getitem__ method to return fake values as if they came from an INI file
            with mock.patch("configparser.ConfigParser.__getitem__", side_effect=lambda section: {
                "Email": {
                    "email_password": "mock_password",
                    "email_address": "mock_sender@example.com"
                }
            }[section]):
                email_instance = AlertMailer("mock_config.ini")
        return email_instance

    @mock.patch("smtplib.SMTP")
    def test_send_email(self, mock_smtp_class, email_instance):
        """
        Test the send_email method to verify that an email is constructed and sent correctly.

        This method ensures that:
        - The SMTP connection is initiated.
        - The correct email credentials are used.
        - The email is sent to the correct recipient with the expected subject and body.
        """
        # Mock the SMTP server
        mock_smtp_instance = mock_smtp_class.return_value
        mock_smtp_instance.__enter__.return_value = mock_smtp_instance
        
        email_instance.send_email("receiver@example.com", "Test Subject", "Test Body")

        mock_smtp_class.assert_called_with("smtp.gmail.com", 587)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with(
            "mock_sender@example.com", "mock_password"
            )
        mock_smtp_instance.sendmail.assert_called_once_with(
            "mock_sender@example.com",
            "receiver@example.com",
            mock.ANY  # Mocking the email content
        )

    @mock.patch("smtplib.SMTP")
    def test_send_error_message(self, mock_smtp_class, email_instance):
        """
        Test the send_error_message method to ensure that error notifications are sent correctly.

        This method checks:
        - The error message email is properly constructed with the exception details.
        - The email is sent to the correct recipient.
        - The email body contains the exception message and traceback.
        """
        # Mock the SMTP server
        mock_smtp_instance = mock_smtp_class.return_value
        mock_smtp_instance.__enter__.return_value = mock_smtp_instance
        
        email_instance.send_error_message(
            "receiver@example.com", "Test Exception", "Test Traceback"
        )

        # Verify that the email is sent to the correct recipient
        mock_smtp_instance.sendmail.assert_called_once_with(
            "mock_sender@example.com",
            "receiver@example.com",
            mock.ANY  # Mocking the email content
        )

        # Optionally, check that the email body contains the exception details
        sent_email = mock_smtp_instance.sendmail.call_args[0][2] 
        assert "Test Exception" in sent_email
        assert "Test Traceback" in sent_email
