import torch
import numpy as np
from PIL import Image

class YOLOv5Detector:
    def __init__(self, model_name="yolov5s", device="cpu", conf_threshold=0.4):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device).eval()
        self.model.conf = conf_threshold # confidence threshold
        self.class_names = self.model.names

    def predict(self, image):
        """
        Gives clean and JSON-friendly output for the detections.
        Takes a PIL image and returns list of detections:
        Each detection is a dict: {
        "bbox": [xmin, ymin, xmax, ymax],
        "confidence": float,
        "class_id": int,
        "class_name": str
        }
        """

        results = self.model(image, size=640) # Forward pass
        detections = results.xyxy[0].cpu().numpy() # Convert to numpy array [x1, y1, x2, y2, conf, cls]

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
        """
        Draws bounding boxes and labels on the image.
        Input:
        - image: PIL image
        - detections: output from self.predict()
        
        Output:
        - Returns a PIL image with detections drawn.
        """
        img_np = np.array(image.copy())
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        return Image.fromarray(img_np)