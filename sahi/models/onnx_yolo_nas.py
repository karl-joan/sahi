import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sahi.models.onnx_model import ONNXDetectionModel

from sahi.utils.onnx_model import non_max_supression, xywh2xyxy

logger = logging.getLogger(__name__)

class ONNXYoloNas(ONNXDetectionModel):
    def _post_process(self, outputs: np.ndarray, input_shape: Tuple[int, int], image_shape: Tuple[int, int]) -> List[torch.Tensor]:
        image_h, image_w = image_shape
        input_w, input_h = input_shape

        boxes = np.squeeze(outputs[0], axis=0)
        predictions = np.squeeze(outputs[1], axis=0)

        # Filter out object confidence scores below threshold
        scores = np.max(predictions, axis=1)
        boxes = boxes[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]
        class_ids = np.argmax(predictions, axis=1)

        # Scale boxes to original dimensions
        input_shape = np.array([input_w, input_h, input_w, input_h])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_w, image_h, image_w, image_h])
        boxes = boxes.astype(np.int32)

        # Perform non-max supressions
        indices = non_max_supression(boxes, scores, self.iou_threshold)

        # Format the results
        prediction_result = []
        for bbox, score, label in zip(boxes[indices], scores[indices], class_ids[indices]):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

        prediction_result = [torch.tensor(prediction_result)]

        return prediction_result