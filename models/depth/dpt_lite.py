# models/depth/dpt_lite.py
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

class DPTLite:
    def __init__(self, device="cpu"):
        self.device = device
        self.model_type = "DPT_Lite"
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def predict(self, image: Image.Image):
        input_tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map
