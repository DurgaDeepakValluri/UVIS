"""
yolov5n.py – Ultra-light YOLOv5n Wrapper
This is almost identical to yolov5.py, 
but loads the YOLOv5n model, 
which is lighter and faster.
"""

import torch
import numpy as np
from PIL import Image
import cv2


class YOLOv5NanoDetector:
    def __init__(self, devide="cpu", conf_threshold=0.4):
        self.device = device,
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.to(self.device).eval()
        self.model.conf = conf_threshold
        self.class_names = self.model.names

    def predict(self, image):
        results = self.model(image, size=640)
        detections = results.xyxy[0].cpu().numpy()

        output = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            output.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": self.class_names[int(cls_id)]
            })

        return output
    
    def draw_dets(self, image, detections):
        img_np = np.array(image.copy())
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        return Image.fromarray(img_np)
