from __future__ import annotations

"""Local preprocessing for text recognition inputs."""

import math

import cv2
import numpy as np

from src.models.text_recognition.config import RecognitionPreprocessConfig


class RecognitionPreprocessor:
    """Crop detection boxes and normalise them for recognition."""

    def __init__(self, config: RecognitionPreprocessConfig) -> None:
        self._channels, self._height, self._width = config.image_shape
        self._default_wh_ratio = self._width / self._height

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes is None or boxes.size == 0:
            return np.empty((0, self._channels, self._height, self._width), dtype=np.float32)

        crops = []
        ratios = []
        for box in boxes:
            crop = self._extract_crop(image, box)
            if crop.size == 0:
                continue
            crops.append(crop)
            h, w = crop.shape[:2]
            if h > 0:
                ratios.append(w / float(h))

        if not crops:
            return np.empty((0, self._channels, self._height, self._width), dtype=np.float32)

        max_wh_ratio = max([self._default_wh_ratio] + ratios)
        processed = []
        for crop in crops:
            norm = self._resize_norm_img(crop, max_wh_ratio)
            processed.append(norm[np.newaxis, :])

        return np.concatenate(processed, axis=0).astype(np.float32)

    def _extract_crop(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        pts = np.array(box, dtype=np.float32)
        h, w = image.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        rect = self._order_points(pts)

        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        height_right = np.linalg.norm(rect[1] - rect[2])
        height_left = np.linalg.norm(rect[0] - rect[3])

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_right, height_left))
        if max_width < 1 or max_height < 1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_c, img_h, img_w = self._channels, self._height, self._width
        img_w = int(max(img_h * max_wh_ratio, 1))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))
        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding[:, :, 0:resized_w] = resized_image
        return padding

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
