from __future__ import annotations

"""Configuration utilities for the text detection stage."""

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.config import ModelIOConfig, ModelOutputConfig, TritonModelConfig


@dataclass(frozen=True)
class DetectionPreprocessConfig:
    """Parameters controlling detection preprocessing."""

    limit_side_len: int
    limit_type: str
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class DetectionPostprocessConfig:
    """Parameters controlling detection postprocessing."""

    thresh: float
    box_thresh: float
    max_candidates: int
    unclip_ratio: float
    use_dilation: bool
    score_mode: str
    box_type: str


@dataclass(frozen=True)
class TextDetectionConfig:
    """Aggregate configuration for the detection pipeline."""

    model: TritonModelConfig
    preprocess: DetectionPreprocessConfig
    postprocess: DetectionPostprocessConfig


def _load_inputs(defs: Iterable[dict]) -> list[ModelIOConfig]:
    inputs: list[ModelIOConfig] = []
    for entry in defs or []:
        name = entry.get("name")
        dtype = entry.get("dtype")
        if not name or not dtype:
            continue
        inputs.append(
            ModelIOConfig(
                name=str(name),
                dtype=str(dtype),
                binary_data=bool(entry.get("binary_data", True)),
            )
        )
    return inputs


def _load_outputs(defs: Iterable[dict]) -> list[ModelOutputConfig]:
    outputs: list[ModelOutputConfig] = []
    for entry in defs or []:
        name = entry.get("name")
        if not name:
            continue
        outputs.append(
            ModelOutputConfig(
                name=str(name),
                binary_data=bool(entry.get("binary_data", True)),
            )
        )
    return outputs


def _load_model_config(cfg: dict) -> TritonModelConfig:
    model_name = cfg.get("name")
    if not model_name:
        raise ValueError("Missing 'name' in detection model configuration.")
    inputs = _load_inputs(cfg.get("inputs", []))
    outputs = _load_outputs(cfg.get("outputs", []))
    return TritonModelConfig(model_name=str(model_name), inputs=inputs, outputs=outputs)


def _to_float_tuple(values: Sequence[float], expected: int = 3) -> tuple[float, ...]:
    seq = list(values)
    if len(seq) < expected:
        seq = seq + [0.0] * (expected - len(seq))
    return tuple(float(v) for v in seq[:expected])


def load_text_detection_config(cfg: dict) -> TextDetectionConfig:
    """Create a :class:`TextDetectionConfig` from a dictionary."""

    model = _load_model_config(cfg.get("model", {}))

    preprocess_cfg = cfg.get("preprocess", {})
    preprocess = DetectionPreprocessConfig(
        limit_side_len=int(preprocess_cfg.get("limit_side_len", 960)),
        limit_type=str(preprocess_cfg.get("limit_type", "max")),
        scale=float(preprocess_cfg.get("scale", 1.0 / 255.0)),
        mean=_to_float_tuple(preprocess_cfg.get("mean", (0.485, 0.456, 0.406))),
        std=_to_float_tuple(preprocess_cfg.get("std", (0.229, 0.224, 0.225))),
    )

    postprocess_cfg = cfg.get("postprocess", {})
    postprocess = DetectionPostprocessConfig(
        thresh=float(postprocess_cfg.get("thresh", 0.3)),
        box_thresh=float(postprocess_cfg.get("box_thresh", 0.6)),
        max_candidates=int(postprocess_cfg.get("max_candidates", 1000)),
        unclip_ratio=float(postprocess_cfg.get("unclip_ratio", 1.5)),
        use_dilation=bool(postprocess_cfg.get("use_dilation", False)),
        score_mode=str(postprocess_cfg.get("score_mode", "fast")),
        box_type=str(postprocess_cfg.get("box_type", "quad")),
    )

    return TextDetectionConfig(model=model, preprocess=preprocess, postprocess=postprocess)
