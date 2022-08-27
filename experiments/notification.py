"""Shoots me a text about something going on in the live demo"""
import os

from flask import Flask
import gin
from twilio.rest import Client

app = Flask(__name__)


@gin.configurable
def load_twilio_keys(filepath=None):
    if filepath is not None:
        with open(filepath, "r") as file:
            data = file.readlines()

        account_sid = data[0].replace("\n", "")
        auth_token = data[1].replace("\n", "")
        set_twilio_env(account_sid, auth_token)

        account_sid = os.environ['TWILIO_ACCOUNT_SID']
        auth_token = os.environ['TWILIO_AUTH_TOKEN']
        client = Client(account_sid, auth_token)
        return client
    else:
        return None


def set_twilio_env(account_sid, auth_token):
    """Sets environment variables"""
    os.environ["TWILIO_ACCOUNT_SID"] = account_sid
    os.environ["TWILIO_AUTH_TOKEN"] = auth_token


@gin.configurable
def notify_myself(message, send_to=None, twilio_number=None, client=None, login_alert=False):
    if send_to is None or client is None or not login_alert:
        app.logger.info(f"No text sent, message logged: {message}")
    else:
        _ = client.messages.create(
            body=message,
            from_=twilio_number,
            to=send_to
        )
        app.logger.info(f"Sent message {message} to phone number {send_to}")
