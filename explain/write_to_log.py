"""A simple logging routine to store inputs to the system."""
from datetime import datetime
from pytz import timezone, utc

import boto3
from flask import Flask
import gin

app = Flask(__name__)


def pst():
    # From https://gist.github.com/vladwa/8cd97099e32c1088025dfaca5f1bfd33
    date_format = '%m_%d_%Y_%H_%M_%S_%Z'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime = date.strftime(date_format)
    return pstDateTime


@gin.configurable
def log_dialogue_input(log_dict, dynamodb_table):
    """Logs dialogue input to file."""
    if not isinstance(log_dict, dict):
        raise NameError(f"Logging information must be dictionary, not type {type(log_dict)}")

    # Log in PST
    log_dict["time"] = pst()
    if dynamodb_table is not None:
        try:
            dynamodb_table.put_item(Item=log_dict)
            app.logger.info("DB write successful")
        except Exception as e:
            app.logger.info(f"Could not write to database: {e}")
    # If no db is specified, write logs to info
    app.logger.info(log_dict)


def load_aws_keys(filepath):
    with open(filepath, "r") as file:
        data = file.readlines()
    return {"access_key": data[0].replace("\n", ""), "secret_key": data[1].replace("\n", "")}


@gin.configurable
def load_dynamo_db(key_filepath, region_name, table_name):
    """Loads dynamo db."""

    if key_filepath is not None:
        keys = load_aws_keys(key_filepath)
        access_key, secret_key = keys["access_key"], keys["secret_key"]
        dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )

        return dynamodb.Table(table_name)
    return None
