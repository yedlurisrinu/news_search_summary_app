"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 12:19 PM
@File: elastic_setup.py
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_PATH = str(Path(__file__).resolve().parent.parent)
def get_config():
    resource_path = BASE_PATH+"/resources/elastic-config.properties"
    config = {}
    try:
        with open(resource_path, 'r') as file_handle:
            entries = file_handle.readlines()
            for entry in entries:
                entry = entry.strip()
                if len(entry) > 0 and not entry.startswith("#"):
                    key, value = entry.split(sep="=", maxsplit=1)
                    config[key.strip()] = value.strip()

    except FileNotFoundError as ex:
        logger.error(" Exception while reading elasticsearch config files: %s ", ex, exc_info=True)

    return config

