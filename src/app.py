from __future__ import annotations

"""FastAPI application exposing the OCR pipeline."""

import os
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import TritonClientConfig
from src.models.text_detection.factory import TextDetectionFactory
from src.models.text_recognition.factory import TextRecognitionFactory
from src.pipeline.ocr import OCRPipeline, OCRPipelineError


class OCRItem(BaseModel):
    box: List[List[float]]
    text: str
    score: float | None = None


class OCRResponse(BaseModel):
    results: List[OCRItem]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _build_pipeline() -> OCRPipeline:
    client_config = TritonClientConfig(
        url=os.getenv("TRITON_URL", "localhost:8001"),
        protocol=os.getenv("TRITON_PROTOCOL", "http"),
        verbose=_env_flag("TRITON_VERBOSE", False),
        timeout=_env_int("TRITON_TIMEOUT"),
        ssl=_env_flag("TRITON_SSL", False),
        root_certificates=os.getenv("TRITON_ROOT_CERTIFICATES"),
        private_key=os.getenv("TRITON_PRIVATE_KEY"),
        certificate_chain=os.getenv("TRITON_CERTIFICATE_CHAIN"),
    )

    config_path = os.getenv("OCR_MODEL_CONFIG")

    detection_model = TextDetectionFactory.get_instance(client_config, config_path=config_path)
    recognition_model = TextRecognitionFactory.get_instance(client_config, config_path=config_path)
    return OCRPipeline(detection_model, recognition_model)


app = FastAPI(title="Triton OCR Service", version="1.0.0")
pipeline = _build_pipeline()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer", response_model=OCRResponse)
async def infer(file: UploadFile = File(...)) -> OCRResponse:
    contents = await file.read()
    await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    try:
        results = pipeline.run(image)
    except OCRPipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for unexpected errors
        raise HTTPException(status_code=500, detail="Unexpected OCR failure.") from exc
    return OCRResponse(results=results)
