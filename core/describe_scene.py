import numpy as np
import logging

logger = logging.getLogger(__name__)

def describe_scene(detection=None, segmentation=None, depth=None):
    """
    Generates a structured scene summary with metrics for detection, segmentation, and depth.

    Args:
        detection (list): List of detected objects with class names and bounding boxes.
        segmentation (numpy.ndarray): Segmentation mask as a 2D numpy array.
        depth (numpy.ndarray): Depth map as a 2D numpy array.

    Returns:
        dict: Structured scene description with metrics.
    """
    logger.info("Generating scene summary...")
    description = {"scene_summary": {}}

    # Detection Summary with Metrics
    if detection:
        logger.info("Adding detection results to scene summary.")
        description["scene_summary"]["objects"] = detection
        confidences = [obj.get("confidence", 0) for obj in detection]
        description["scene_summary"]["detection_metrics"] = {
            "objects_detected": len(detection),
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0
        }

    # Segmentation Summary with Coverage Metrics
    if segmentation is not None:
        logger.info("Summarizing segmentation coverage.")
        unique, counts = np.unique(segmentation, return_counts=True)
        total = segmentation.size
        coverage = [
            {"class_id": int(class_id), "coverage": f"{(count / total) * 100:.2f}%"}
            for class_id, count in zip(unique, counts)
        ]
        dominant_class = max(coverage, key=lambda x: float(x["coverage"].strip('%')))
        description["scene_summary"]["segmentation_summary"] = coverage
        description["scene_summary"]["dominant_class"] = dominant_class

    # Depth Summary with Metrics
    if depth is not None:
        logger.info("Summarizing depth information.")
        mean_depth = float(np.mean(depth))
        min_depth = float(np.min(depth))
        max_depth = float(np.max(depth))
        std_depth = float(np.std(depth))
        description["scene_summary"]["depth_summary"] = {
            "mean_depth": mean_depth,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "std_depth": std_depth
        }

    logger.info("Scene summary generation complete.")
    return description
