from __future__ import annotations

"""Factory utilities for the text recognition model."""

from pathlib import Path
from typing import Optional

from src.config import TritonClientConfig
from src.models.configuration import load_ocr_models_config
from src.models.text_recognition.config import TextRecognitionConfig
from src.models.text_recognition.model import TextRecognitionModel


class TextRecognitionFactory:
    """Singleton factory for :class:`TextRecognitionModel`."""

    _instance: TextRecognitionModel | None = None

    @classmethod
    def get_instance(
        cls,
        client_config: TritonClientConfig,
        config: Optional[TextRecognitionConfig] = None,
        config_path: str | Path | None = None,
    ) -> TextRecognitionModel:
        if cls._instance is None:
            recognition_config = config
            if recognition_config is None:
                models_config = load_ocr_models_config(None if config_path in (None, "") else config_path)
                recognition_config = models_config.recognition
            cls._instance = TextRecognitionModel(client_config=client_config, config=recognition_config)
        return cls._instance
