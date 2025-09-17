from __future__ import annotations

"""Configuration utilities for the text recognition stage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.config import ModelIOConfig, ModelOutputConfig, TritonModelConfig


@dataclass(frozen=True)
class RecognitionPreprocessConfig:
    """Parameters for recognition preprocessing."""

    image_shape: tuple[int, int, int]


@dataclass(frozen=True)
class RecognitionPostprocessConfig:
    """Parameters for recognition postprocessing."""

    dictionary_path: Path | None
    use_space_char: bool


@dataclass(frozen=True)
class TextRecognitionConfig:
    """Aggregate configuration for the recognition pipeline."""

    model: TritonModelConfig
    preprocess: RecognitionPreprocessConfig
    postprocess: RecognitionPostprocessConfig


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
        raise ValueError("Missing 'name' in recognition model configuration.")
    inputs = _load_inputs(cfg.get("inputs", []))
    outputs = _load_outputs(cfg.get("outputs", []))
    return TritonModelConfig(model_name=str(model_name), inputs=inputs, outputs=outputs)


def _to_int_tuple(values: Sequence[int], expected: int = 3) -> tuple[int, ...]:
    seq = list(values)
    if len(seq) < expected:
        seq = seq + [0] * (expected - len(seq))
    return tuple(int(v) for v in seq[:expected])


def load_text_recognition_config(cfg: dict, base_path: Path | None = None) -> TextRecognitionConfig:
    """Create a :class:`TextRecognitionConfig` from a dictionary."""

    model = _load_model_config(cfg.get("model", {}))

    preprocess_cfg = cfg.get("preprocess", {})
    preprocess = RecognitionPreprocessConfig(
        image_shape=_to_int_tuple(preprocess_cfg.get("image_shape", (3, 48, 320))),
    )

    postprocess_cfg = cfg.get("postprocess", {})
    dict_path_value = postprocess_cfg.get("dictionary_path")
    dictionary_path: Path | None
    if dict_path_value:
        dictionary_path = Path(dict_path_value)
        if base_path and not dictionary_path.is_absolute():
            dictionary_path = (base_path / dictionary_path).resolve()
    else:
        dictionary_path = None
    postprocess = RecognitionPostprocessConfig(
        dictionary_path=dictionary_path,
        use_space_char=bool(postprocess_cfg.get("use_space_char", True)),
    )

    return TextRecognitionConfig(model=model, preprocess=preprocess, postprocess=postprocess)
