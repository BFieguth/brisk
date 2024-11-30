"""Provides the AlertMailer class for sending notification and error emails.

Exports:
    - AlertMailer: A class to send emails using Gmail's SMTP service.
"""

import configparser
import smtplib
from email.mime import text, multipart

class AlertMailer:
    """A class to send notification emails.

    This class handles sending emails using Gmail as the email service
    provider. It is initialized with configuration from a config file
    containing the sender"s email and password.

    Attributes:
        sender_password (str): The password for the sender"s email account.
        sender_email (str): The sender"s email address.
    """

    def __init__(self, config_path: str):
        """Initializes AlertMailer with email credentials.

        Args:
            config_path (str): The path to the configuration file containing
                the email credentials.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        self.sender_password: str = config["Email"]["email_password"]
        self.sender_email: str = config["Email"]["email_address"]

    def send_email(
        self,
        receiver_email: str,
        subject: str,
        body: str
    ) -> None:
        """Send an email using SMTP.

        Args:
            receiver_email (str): The recipient"s email address.
            subject (str): The subject line of the email.
            body (str): The body content of the email.
        """
        message = multipart.MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(text.MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(
                self.sender_email, receiver_email, message.as_string()
                )

    def send_error_message(
        self,
        receiver_email: str,
        exception: str,
        traceback: str
    ) -> None:
        """Sends an error notification email.

        Args:
            receiver_email (str): The recipient"s email address.
            exception (str): The exception message.
            traceback (str): The traceback details of the exception.
        """
        subject = "Error Notification"
        body = (f"An error occurred during execution:\n\nError Message:\n"
                f"{exception}\n\nTraceback:\n{traceback}")
        self.send_email(receiver_email, subject, body)
