import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
from models.segmentation.color_map_cityscapes import COLOR_MAP as CITYSCAPES_COLOR_MAP
from models.segmentation.label_map_cityscapes import LABEL_MAP as CITYSCAPES_LABEL_MAP
from models.segmentation.color_map_ade20k import COLOR_MAP as ADE20K_COLOR_MAP
from models.segmentation.label_map_ade20k import LABEL_MAP as ADE20K_LABEL_MAP

from models.segmentation.bisenetv2_core import BiSeNetV2

class BiSeNetV2Wrapper:
    def __init__(self, dataset="cityscapes", weight_path="models/segmentation/bisenetv2_cityscapes.pth", device="cpu"):
        self.device = device
        self.model = BiSeNetV2(n_classes=19 if dataset == "cityscapes" else 150)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.to(device).eval()

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
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
            out = self.model(img)[0]  # [C, H, W]
            pred = out.argmax(0).cpu().numpy()

        return pred

    def draw(self, image, mask, alpha=0.5):
        # Resize mask to image size
        mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

        # Color all classes using color_map
        for class_id, color in self.color_map.items():
            color_mask[mask_resized == class_id] = color

        img_np = np.array(image)
        overlay = cv2.addWeighted(img_np, 1 - alpha, color_mask, alpha, 0)

        return Image.fromarray(overlay)
