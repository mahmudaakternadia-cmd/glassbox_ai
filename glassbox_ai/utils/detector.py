import cv2
import numpy as np
from ultralytics import YOLO
import torch


class ObjectDetector:
    """YOLOv8 Object Detector — wraps Ultralytics for clean inference."""

    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    ]

    def __init__(self, config: dict):
        self.config = config
        model_name = config['model']['name']
        device_pref = config['model'].get('device', 'auto')

        if device_pref == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_pref

        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, image_bgr: np.ndarray,
               confidence_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> dict:
        """
        Run inference on a BGR numpy image.

        Returns
        -------
        dict with keys:
            boxes        – list of [x1, y1, x2, y2]
            confidences  – list of float (0–1)
            class_ids    – list of int
            class_names  – list of str
        """
        results = self.model(
            image_bgr,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: dict = {
            'boxes': [], 'confidences': [], 'class_ids': [], 'class_names': []
        }

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf   = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections['boxes'].append([x1, y1, x2, y2])
                detections['confidences'].append(conf)
                detections['class_ids'].append(cls_id)
                detections['class_names'].append(self.COCO_CLASSES[cls_id])

        return detections

    @property
    def class_names(self):
        return self.COCO_CLASSES