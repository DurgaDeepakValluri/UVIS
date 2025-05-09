# models/depth/midas_small.py
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

class MiDaSSmall:
    def __init__(self, device="cpu"):
        self.device = device
        self.model_type = "DPT_Small"  # MiDaS v3 - small
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def predict(self, image: Image.Image):
        input_tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map
