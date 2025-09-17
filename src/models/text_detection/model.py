from __future__ import annotations

"""Text detection model wrapper."""

import numpy as np

from src.config import TritonClientConfig
from src.models.text_detection.config import TextDetectionConfig
from src.models.text_detection.postprocess import DetectionPostprocessor
from src.models.text_detection.preprocess import DetectionPreprocessor
from src.triton.base import BaseTritonClient, TritonClientError


class TextDetectionError(RuntimeError):
    """Raised when the text detection stage fails."""


class TextDetectionModel(BaseTritonClient):
    """Execute the text detection stage using Triton served models."""

    def __init__(self, client_config: TritonClientConfig, config: TextDetectionConfig) -> None:
        super().__init__(client_config)
        self._config = config
        self._preprocessor = DetectionPreprocessor(config.preprocess)
        self._postprocessor = DetectionPostprocessor(config.postprocess)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run detection and return the detected quadrilateral boxes."""

        if image.ndim != 3 or image.shape[2] != 3:
            raise TextDetectionError("The detection model expects a color image with shape (H, W, 3).")

        try:
            processed_image, shape_list = self._preprocessor(image)
        except Exception as exc:
            raise TextDetectionError(f"Detection preprocessing failed: {exc}") from exc

        if not self._config.model.inputs:
            raise TextDetectionError("Detection model configuration must define at least one input.")
        input_name = self._config.model.inputs[0].name
        model_inputs = {input_name: processed_image.astype(np.float32, copy=False)}

        try:
            detection_outputs = self._infer(self._config.model, model_inputs)
        except TritonClientError as exc:
            raise TextDetectionError(f"Detection inference request failed: {exc}") from exc
        except Exception as exc:
            raise TextDetectionError(f"Unexpected error during detection inference: {exc}") from exc

        try:
            if self._config.model.outputs:
                output_name = self._config.model.outputs[0].name
                detection_map = detection_outputs[output_name]
            else:
                # Fallback for models without explicit output configuration.
                detection_map = next(iter(detection_outputs.values()))
        except (KeyError, StopIteration) as exc:
            raise TextDetectionError("Detection outputs missing expected tensors.") from exc

        try:
            boxes = self._postprocessor(detection_map.astype(np.float32, copy=False), shape_list)
        except Exception as exc:
            raise TextDetectionError(f"Detection postprocessing failed: {exc}") from exc
        return boxes
