from __future__ import annotations

"""Helpers to load OCR model configuration from YAML."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from src.models.text_detection.config import TextDetectionConfig, load_text_detection_config
from src.models.text_recognition.config import TextRecognitionConfig, load_text_recognition_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class OCRModelsConfig:
    """Group detection and recognition configuration."""

    detection: TextDetectionConfig
    recognition: TextRecognitionConfig


@lru_cache(maxsize=None)
def load_ocr_models_config(config_path: str | Path | None = None) -> OCRModelsConfig:
    """Load detection and recognition configuration from a YAML file."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        raw: Dict[str, Any] = yaml.safe_load(stream) or {}

    detection_cfg = load_text_detection_config(raw.get("detection", {}))
    recognition_cfg = load_text_recognition_config(raw.get("recognition", {}), base_path=path.parent)

    return OCRModelsConfig(detection=detection_cfg, recognition=recognition_cfg)
