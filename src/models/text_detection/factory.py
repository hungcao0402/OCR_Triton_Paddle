from __future__ import annotations

"""Factory utilities for the text detection model."""

from pathlib import Path
from typing import Optional

from src.config import TritonClientConfig
from src.models.configuration import load_ocr_models_config
from src.models.text_detection.config import TextDetectionConfig
from src.models.text_detection.model import TextDetectionModel


class TextDetectionFactory:
    """Singleton factory for :class:`TextDetectionModel`."""

    _instance: TextDetectionModel | None = None

    @classmethod
    def get_instance(
        cls,
        client_config: TritonClientConfig,
        config: Optional[TextDetectionConfig] = None,
        config_path: str | Path | None = None,
    ) -> TextDetectionModel:
        if cls._instance is None:
            detection_config = config
            if detection_config is None:
                models_config = load_ocr_models_config(None if config_path in (None, "") else config_path)
                detection_config = models_config.detection
            cls._instance = TextDetectionModel(client_config=client_config, config=detection_config)
        return cls._instance
