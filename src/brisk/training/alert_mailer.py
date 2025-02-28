"""Provides the AlertMailer class for sending notification and error emails.

This module provides functionality for sending email notifications using Gmail's
SMTP service.
"""

import configparser
import smtplib
from email.mime import text, multipart

class AlertMailer:
    """A class to send notification emails using Gmail's SMTP service.

    Parameters
    ----------
    config_path : str
        Path to configuration file containing email credentials

    Attributes
    ----------
    sender_password : str
        Password for the sender's email account
    sender_email : str
        Sender's email address

    Notes
    -----
    The configuration file should be in INI format with an [Email] section
    containing email_password and email_address fields. 
    """

    def __init__(self, config_path: str):
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

        Parameters
        ----------
        receiver_email : str
            Recipient's email address
        subject : str
            Subject line of the email
        body : str
            Body content of the email

        Raises
        ------
        smtplib.SMTPAuthenticationError
            If authentication with Gmail fails
        smtplib.SMTPException
            If there's an error sending the email
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
        """Send an error notification email.

        Parameters
        ----------
        receiver_email : str
            Recipient's email address
        exception : str
            Exception message
        traceback : str
            Traceback details of the exception

        Notes
        -----
        The email will be formatted with a standard subject line
        "Error Notification" and will include both the exception
        message and traceback in the body.
        """
        subject = "Error Notification"
        body = (f"An error occurred during execution:\n\nError Message:\n"
                f"{exception}\n\nTraceback:\n{traceback}")
        self.send_email(receiver_email, subject, body)
