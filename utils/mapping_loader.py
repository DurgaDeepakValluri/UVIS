import logging

logger = logging.getLogger(__name__)

def load_color_and_label_maps(dataset):
    """
    Dynamically loads color and label maps based on the dataset.

    Args:
        dataset (str): The dataset name ("cityscapes" or "ade20k").

    Returns:
        tuple: COLOR_MAP and LABEL_MAP dictionaries.

    Raises:
        ValueError: If the dataset is not supported.
    """
    logger.info(f"Loading color and label maps for dataset: {dataset}")
    if dataset == "cityscapes":
        from models.segmentation.color_map_cityscapes import COLOR_MAP, LABEL_MAP
    elif dataset == "ade20k":
        from models.segmentation.color_map_ade20k import COLOR_MAP, LABEL_MAP
    else:
        logger.error(f"Unsupported dataset requested: {dataset}")
        raise ValueError(f"Unsupported dataset: {dataset}")
    logger.info(f"Maps loaded successfully for dataset: {dataset}")
    return COLOR_MAP, LABEL_MAP
