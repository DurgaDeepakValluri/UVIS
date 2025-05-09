import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_array(arr):
    """
    Normalizes a numpy array to the range [0, 1].

    Args:
        arr (numpy.ndarray): The array to normalize.

    Returns:
        numpy.ndarray: The normalized array.
    """
    logger.info("Normalizing array to range [0, 1].")
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
