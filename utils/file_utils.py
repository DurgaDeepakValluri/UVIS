import os
import logging

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """
    Ensures the given directory exists. Creates it if it does not.

    Args:
        directory (str): The directory path to check or create.
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory)
    else:
        logger.info(f"Directory already exists: {directory}")
