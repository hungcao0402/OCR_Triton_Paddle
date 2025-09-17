from __future__ import annotations

"""Composable OCR pipeline built on top of the Triton inference server."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.models.text_detection.model import TextDetectionError, TextDetectionModel
from src.models.text_recognition.model import TextRecognitionError, TextRecognitionModel


@dataclass
class OCRResult:
    """Container for a single OCR prediction."""

    box: List[List[float]]
    text: str
    score: float | None

    def as_dict(self) -> Dict[str, Any]:
        return {"box": self.box, "text": self.text, "score": self.score}


class OCRPipelineError(RuntimeError):
    """Raised when a stage of the OCR pipeline fails."""


logger = logging.getLogger(__name__)


class OCRPipeline:
    """Run text detection followed by recognition and merge their outputs."""

    def __init__(self, detection_model: TextDetectionModel, recognition_model: TextRecognitionModel) -> None:
        self._detection_model = detection_model
        self._recognition_model = recognition_model

    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return self.run(image)

    def run(self, image: np.ndarray) -> List[Dict[str, Any]]:
        logger.debug("Running detection stage.")
        try:
            boxes = self._detection_model.predict(image)
        except TextDetectionError as exc:
            raise OCRPipelineError(f"Text detection failed: {exc}") from exc
        except Exception as exc:
            raise OCRPipelineError(f"Text detection failed: {exc}") from exc
        if isinstance(boxes, np.ndarray):
            num_boxes = 0 if boxes.size == 0 else int(boxes.shape[0])
            if boxes.size == 0:
                return []
        else:
            num_boxes = len(boxes)
            if not boxes:
                return []
        logger.info("Detection stage produced %d candidate box(es).", num_boxes)

        logger.debug("Running recognition stage for %d box(es).", num_boxes)
        try:
            texts, scores = self._recognition_model.predict(image, boxes)
        except TextRecognitionError as exc:
            raise OCRPipelineError(f"Text recognition failed: {exc}") from exc
        except Exception as exc:
            raise OCRPipelineError(f"Text recognition failed: {exc}") from exc
        results: List[OCRResult] = []
        for idx in range(num_boxes):
            box = boxes[idx]
            text = texts[idx] if idx < len(texts) else ""
            score = float(scores[idx]) if idx < len(scores) else None
            results.append(
                OCRResult(
                    box=box.astype(float).tolist() if isinstance(box, np.ndarray) else box,
                    text=text,
                    score=score,
                )
            )
        logger.info("Recognition stage produced %d OCR result(s).", len(results))
        return [result.as_dict() for result in results]
