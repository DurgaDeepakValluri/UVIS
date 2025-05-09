import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2  # You missed importing this

from models.segmentation.color_map_cityscapes import COLOR_MAP as CITYSCAPES_COLOR_MAP
from models.segmentation.label_map_cityscapes import LABEL_MAP as CITYSCAPES_LABEL_MAP
from models.segmentation.color_map_ade20k import COLOR_MAP as ADE20K_COLOR_MAP
from models.segmentation.label_map_ade20k import LABEL_MAP as ADE20K_LABEL_MAP

class FastSeg:
    def __init__(self, dataset="cityscapes", device="cpu"):
        self.device = device
        self.model = torch.hub.load(
            "ekzhang/fastseg", "fastseg_mobilenetv3", pretrained=True
        ).to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        if dataset == "cityscapes":
            self.color_map = CITYSCAPES_COLOR_MAP
            self.label_map = CITYSCAPES_LABEL_MAP
        else:
            self.color_map = ADE20K_COLOR_MAP
            self.label_map = ADE20K_LABEL_MAP

    def predict(self, image):
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img)[0]  # shape: [num_classes, H, W]
            mask = pred.argmax(0).byte().cpu().numpy()  # shape: [H, W]
        return mask

    def draw(self, image, mask, alpha=0.5):
        # Resize mask to image size
        mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)

        # Build color overlay
        color_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            color_mask[mask_resized == class_id] = color

        # Blend overlay with original image
        img_np = np.array(image)
        overlay = cv2.addWeighted(img_np, 1 - alpha, color_mask, alpha, 0)

        return Image.fromarray(overlay)
