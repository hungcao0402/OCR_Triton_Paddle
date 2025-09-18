from __future__ import annotations

"""Text recognition model wrapper."""

import logging
from time import perf_counter

import numpy as np

from src.config import TritonClientConfig
from src.models.text_recognition.config import TextRecognitionConfig
from src.models.text_recognition.postprocess import RecognitionPostprocessor
from src.models.text_recognition.preprocess import RecognitionPreprocessor
from src.triton.base import BaseTritonClient, TritonClientError


class TextRecognitionError(RuntimeError):
    """Raised when the text recognition stage fails."""


logger = logging.getLogger(__name__)


class TextRecognitionModel(BaseTritonClient):
    """Execute the text recognition stage using Triton served models."""

    def __init__(self, client_config: TritonClientConfig, config: TextRecognitionConfig) -> None:
        super().__init__(client_config)
        self._config = config
        self._preprocessor = RecognitionPreprocessor(config.preprocess)
        self._postprocessor = RecognitionPostprocessor(config.postprocess)

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Run recognition on the detected boxes."""

        if boxes is None or getattr(boxes, "size", 0) == 0:
            return [], np.array([], dtype=np.float32)

        logger.debug("Preparing %d detection box(es) for recognition.", len(boxes) if boxes is not None else 0)

        try:
            crops = self._preprocessor(image, boxes)
        except Exception as exc:
            raise TextRecognitionError(f"Recognition preprocessing failed: {exc}") from exc
        if crops.size == 0:
            return [], np.array([], dtype=np.float32)

        if not self._config.model.inputs:
            raise TextRecognitionError("Recognition model configuration must define at least one input.")

        input_name = self._config.model.inputs[0].name
        batch_size = max(1, int(self._config.batch_size))
        num_crops = int(crops.shape[0])
        total_infer_time = 0.0
        batch_count = 0
        all_texts: list[str] = []
        all_scores: list[float] = []

        for start_idx in range(0, num_crops, batch_size):
            batch = crops[start_idx : start_idx + batch_size]
            batch_count += 1
            infer_start = perf_counter()
            try:
                recognition_outputs = self._infer(
                    self._config.model,
                    {input_name: batch.astype(np.float32, copy=False)},
                )
            except TritonClientError as exc:
                raise TextRecognitionError(f"Recognition inference request failed: {exc}") from exc
            except Exception as exc:
                raise TextRecognitionError(f"Unexpected error during recognition inference: {exc}") from exc
            batch_time = perf_counter() - infer_start
            total_infer_time += batch_time

            try:
                if self._config.model.outputs:
                    output_name = self._config.model.outputs[0].name
                    logits = recognition_outputs[output_name]
                else:
                    logits = next(iter(recognition_outputs.values()))
            except (KeyError, StopIteration) as exc:
                raise TextRecognitionError("Recognition outputs missing expected tensors.") from exc

            try:
                batch_texts, batch_scores = self._postprocessor(logits.astype(np.float32, copy=False))
            except Exception as exc:
                raise TextRecognitionError(f"Recognition postprocessing failed: {exc}") from exc

            all_texts.extend(batch_texts)
            all_scores.extend(batch_scores)

        logger.info(
            "Recognition model inference completed in %.2f ms across %d batch(es).",
            total_infer_time * 1000,
            batch_count,
        )
        logger.debug("Recognition post-processing generated %d text prediction(s).", len(all_texts))
        return all_texts, np.array(all_scores, dtype=np.float32)
