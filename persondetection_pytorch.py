# persondetection_pytorch.py

import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class DetectorAPI:
    def __init__(self, threshold=0.7, device=None):
        """
        threshold: minimum score for keeping detection
        device: torch.device or string, e.g. 'cuda' or 'cpu'
        """
        # choose device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # load a pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        # transform: OpenCV BGR image → RGB tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.threshold = threshold

    def processFrame(self, image: np.ndarray):
        """
        image: BGR numpy array (HxWx3) from cv2
        returns: boxes, scores, classes, num_detections
        boxes: list of (y1, x1, y2, x2)
        classes: list of int
        scores: list of float
        num_detections: int
        """
        # convert to RGB and to tensor
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            output = self.model([tensor])[0]

        # pull out tensors, move to CPU & numpy
        boxes   = output["boxes"].cpu().numpy()
        scores  = output["scores"].cpu().numpy()
        labels  = output["labels"].cpu().numpy()

        # filter by threshold
        keep = scores >= self.threshold
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # convert to the (y1, x1, y2, x2) int format your GUI code expects
        h, w = image.shape[:2]
        boxes_list = []
        for (x1, y1, x2, y2) in boxes:
            # clamp and cast
            y1c, x1c = int(max(0, y1)), int(max(0, x1))
            y2c, x2c = int(min(h, y2)), int(min(w, x2))
            boxes_list.append((y1c, x1c, y2c, x2c))

        return boxes_list, scores.tolist(), labels.tolist(), len(boxes_list)

    def close(self):
        # no session to close in PyTorch
        pass
