import json
from src.constants import STD_LOGGING_FORMAT
import logging
from yacs.config import load_cfg


def read_json(file_path: str) -> dict:
    """Reads json file from the given path.

    Args:
        file_path (str): Location of the file

    Returns:
        dict: Json content formatted as python dictionary in most cases
    """
    with open(file_path, "r") as f:
        return json.load(f)


def get_console_logger(script_name: str) -> logging.Logger:
    logger = logging.getLogger(script_name)
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(STD_LOGGING_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_json(obj: dict, file_path: str):
    """Saves the object as a json at given path"""
    with open(file_path, "w") as f:
        json.dump(obj, f)


def read_yaml(filepath: str):
    with open(filepath, "r") as f:
        return load_cfg(f)
