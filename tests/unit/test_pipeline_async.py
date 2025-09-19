import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.ocr import OCRPipeline


class AsyncDetectionStub:
    def __init__(self) -> None:
        self.async_called = False

    async def apredict(self, image: np.ndarray) -> np.ndarray:
        self.async_called = True
        await asyncio.sleep(0)
        return np.array(
            [
                [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 5.0],
                    [0.0, 5.0],
                ]
            ],
            dtype=np.float32,
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        raise AssertionError("Synchronous predict should not be called when async version exists.")


class AsyncRecognitionStub:
    def __init__(self) -> None:
        self.async_called = False

    async def apredict(self, image: np.ndarray, boxes: np.ndarray) -> tuple[list[str], np.ndarray]:
        self.async_called = True
        await asyncio.sleep(0)
        return ["async-text"], np.array([0.9], dtype=np.float32)

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> tuple[list[str], np.ndarray]:
        raise AssertionError("Synchronous predict should not be called when async version exists.")


class SyncDetectionStub:
    def __init__(self) -> None:
        self.called = False

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.called = True
        return np.array(
            [
                [
                    [2.0, 1.0],
                    [4.0, 1.0],
                    [4.0, 3.0],
                    [2.0, 3.0],
                ]
            ],
            dtype=np.float32,
        )


class SyncRecognitionStub:
    def __init__(self) -> None:
        self.called = False
        self.received_boxes: Any = None

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> tuple[list[str], np.ndarray]:
        self.called = True
        self.received_boxes = boxes
        return ["sync-text"], np.array([0.5], dtype=np.float32)


def test_pipeline_arun_prefers_async_models() -> None:
    pipeline = OCRPipeline(AsyncDetectionStub(), AsyncRecognitionStub())
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    results = asyncio.run(pipeline.arun(image))

    assert len(results) == 1
    result = results[0]
    assert result["text"] == "async-text"
    assert result["box"] == [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 5.0],
        [0.0, 5.0],
    ]
    assert result["score"] == pytest.approx(0.9)


def test_pipeline_arun_falls_back_to_sync_models() -> None:
    detection = SyncDetectionStub()
    recognition = SyncRecognitionStub()
    pipeline = OCRPipeline(detection, recognition)
    image = np.zeros((6, 6, 3), dtype=np.uint8)

    results = asyncio.run(pipeline.arun(image))

    assert detection.called is True
    assert recognition.called is True
    assert recognition.received_boxes is not None
    np.testing.assert_array_equal(
        recognition.received_boxes,
        np.array(
            [
                [
                    [2.0, 1.0],
                    [4.0, 1.0],
                    [4.0, 3.0],
                    [2.0, 3.0],
                ]
            ],
            dtype=np.float32,
        ),
    )
    assert len(results) == 1
    result = results[0]
    assert result["box"] == [
        [2.0, 1.0],
        [4.0, 1.0],
        [4.0, 3.0],
        [2.0, 3.0],
    ]
    assert result["text"] == "sync-text"
    assert result["score"] == pytest.approx(0.5)
