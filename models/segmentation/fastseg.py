"""
fastseg.py
FastSeg model for semantic segmentation
Loads the FastSeg segmentation model (MobileNetV3)
Takes a PIL image
Returns a binary/person mask (semantic segmentation)
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class FastSeg:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = torch.hub.load(
            "ekzhang/fastseg", "fastseg_mobilenetv3", pretrained=True
        ).to.(device).eval()
    
        self.transform = T.Compose([
            T.Resize((320, 320)), #Fastseg expectes 320x320 input size
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def predict(self, image):
        """
        Input: PIL image
        Output: mask as a numpy array (binary segmentation map)
        """

        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img)[0] # shape: [num_classes, H, W]
            mask = pred.argmax(0).byte.cpu().numpy() # shape: [H, W]


        return mask