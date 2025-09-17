from __future__ import annotations

"""Composable OCR pipeline built on top of the Triton inference server."""

import logging
from dataclasses import dataclass
from time import perf_counter
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
        pipeline_start = perf_counter()

        logger.debug("Running detection stage.")
        detection_start = perf_counter()
        try:
            boxes = self._detection_model.predict(image)
        except TextDetectionError as exc:
            raise OCRPipelineError(f"Text detection failed: {exc}") from exc
        except Exception as exc:
            raise OCRPipelineError(f"Text detection failed: {exc}") from exc
        detection_time = perf_counter() - detection_start

        if isinstance(boxes, np.ndarray):
            num_boxes = 0 if boxes.size == 0 else int(boxes.shape[0])
        else:
            num_boxes = len(boxes)

        logger.info(
            "Detection stage completed in %.2f ms with %d candidate box(es).",
            detection_time * 1000,
            num_boxes,
        )

        if num_boxes == 0:
            total_time = perf_counter() - pipeline_start
            logger.info(
                "OCR pipeline finished in %.2f ms (detection=%.2f ms, recognition=0.00 ms).",
                total_time * 1000,
                detection_time * 1000,
            )
            return []

        logger.debug("Running recognition stage for %d box(es).", num_boxes)
        recognition_start = perf_counter()
        try:
            texts, scores = self._recognition_model.predict(image, boxes)
        except TextRecognitionError as exc:
            raise OCRPipelineError(f"Text recognition failed: {exc}") from exc
        except Exception as exc:
            raise OCRPipelineError(f"Text recognition failed: {exc}") from exc
        recognition_time = perf_counter() - recognition_start

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
        logger.info(
            "Recognition stage completed in %.2f ms with %d OCR result(s).",
            recognition_time * 1000,
            len(results),
        )

        total_time = perf_counter() - pipeline_start
        logger.info(
            "OCR pipeline finished in %.2f ms (detection=%.2f ms, recognition=%.2f ms).",
            total_time * 1000,
            detection_time * 1000,
            recognition_time * 1000,
        )
        return [result.as_dict() for result in results]
