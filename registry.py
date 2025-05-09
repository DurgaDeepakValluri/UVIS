from models.depth.dpt_lite import DPTLite
from models.depth.midas_small import MiDaSSmall

from models.detection.yolov5 import YOLOv5Detector
from models.detection.yolov5n import YOLOv5NanoDetector

from models.segmentation.bisenetv2 import BiSeNetV2Wrapper
from models.segmentation.fastseg import FastSeg

MODEL_REGISTRY = {
    "detection": {
        "YOLOv5": YOLOv5Detector,
        "YOLOv5Nano": YOLOv5NanoDetector,
    },
    "segmentation": {
        "BiSeNetV2": BiSeNetV2Wrapper,
        "FastSeg": FastSeg,
    },
    "depth": {
        "MiDaSSmall": MiDaSSmall,
        "DPTLite": DPTLite,
    }
}

def get_model(task: str, model_name: str, **kwargs):
    """
    Get a model from the registry.

    Args:
        task (str): The task type (e.g., "detection", "segmentation", "depth").
        model_name (str): The name of the model.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        An instance of the requested model.
    """
    try:
        model_class = MODEL_REGISTRY[task][model_name]
        return model_class(**kwargs)
    except KeyError:
        raise ValueError(f"Model {model_name} for task {task} not found in registry.")