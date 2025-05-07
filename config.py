"""
config.py
Configuration file for the project.
Global settings hub for UVIS. Lets you quickly switch models, adjust thresholds, and define runtime behavior without changing any of the core logic.
"""

import torch

# Determine if GPU is available and if not set device as cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Main config dictionary

CONFIG = {

    # Model selection for each task
    "detection_model": "yolov5s", # Lightweight and fast
    "segmentation_model": "fastseg", # Tiny and Render-COmpatible
    "depth_model": "midas_small", # CPU-friendly

    # Device
    "device": DEVICE,

    # Image Processing
    "image_size": (640, 640), # Resize images to this size for model input
    "conf_threshold": 0.4, # Detection confidence threshold

    # Output Controls
    "save_json": True, # Save structured scene description
    "save_png": True, # Save visual output with overlays
    "save_depth": True, # Save depth map as PNG
    "save_masks": True, # Save segmentation masks as PNG

    # Color mapping and visuals
    "color_palette": "assets/color_palette.json",

    # Deployent settings
    "max_image_size": (640, 640), # To ensure input images are within bounds
}






