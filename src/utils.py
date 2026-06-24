"""
src/utils.py
------------
Common utility functions used across the project, e.g. reading
YAML config files into plain dictionaries.
"""

import sys
import yaml
from src.logger import logging
from src.exception import ChatbotException


def read_yaml(path: str) -> dict:
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        logging.info(f"YAML file loaded successfully from: {path}")
        return content
    except Exception as e:
        raise ChatbotException(e, sys)
