from __future__ import annotations

"""Local preprocessing pipeline for text detection."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.models.text_detection.config import DetectionPreprocessConfig


class DetectionPreprocessor:
    """Prepare raw images for the detection model."""

    def __init__(self, config: DetectionPreprocessConfig) -> None:
        self._resize = DetResizeForTest(limit_side_len=config.limit_side_len, limit_type=config.limit_type)
        self._normalize = NormalizeImage(scale=config.scale, mean=config.mean, std=config.std, order="hwc")
        self._to_chw = ToCHWImage()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Detection preprocessing expects a color image with shape (H, W, 3).")

        data = {"image": image}
        data = self._resize(data)
        data = self._normalize(data)
        data = self._to_chw(data)

        processed_image = data["image"].astype(np.float32)
        shape_list = data["shape"].astype(np.float32)

        return processed_image[np.newaxis, ...], shape_list[np.newaxis, ...]


@dataclass
class DetResizeForTest:
    limit_side_len: int
    limit_type: str = "max"

    def __call__(self, data: dict) -> dict:
        img = data["image"]
        src_h, src_w, _ = img.shape

        img, (ratio_h, ratio_w) = self._resize_image(img)
        data["image"] = img
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
        return data

    def _resize_image(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        elif self.limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise ValueError(f"Unsupported limit_type: {self.limit_type}")

        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)

        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, (ratio_h, ratio_w)


@dataclass
class NormalizeImage:
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    order: str = "hwc"

    def __call__(self, data: dict) -> dict:
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        if self.order not in {"hwc", "chw"}:
            raise ValueError("Order must be either 'hwc' or 'chw'.")
        scale = np.float32(self.scale)
        mean = np.array(self.mean, dtype=np.float32).reshape((1, 1, 3))
        std = np.array(self.std, dtype=np.float32).reshape((1, 1, 3))
        data["image"] = (img.astype(np.float32) * scale - mean) / std
        return data


class ToCHWImage:
    def __call__(self, data: dict) -> dict:
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data
