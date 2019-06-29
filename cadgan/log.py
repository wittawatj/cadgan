"""
A module providing methods for logging.
"""

__author__ = "wittawat"

import logging


# the default logger
def get_logger():
    return logging.getLogger("cadgan_default")


def l():
    return get_logger()


# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format="%(levelname)s: %(asctime)s: %(module)s.%(funcName)s(): %(message)s", level=logging.INFO)

logger = get_logger()
logger.setLevel(logging.DEBUG)
