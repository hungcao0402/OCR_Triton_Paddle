"""Locust load test for the Triton OCR FastAPI service."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import ClassVar

from locust import HttpUser, between, task


def _resolve_sample_image() -> Path:
    """Return the image used for inference requests.

    The path can be customised with the ``LOCUST_SAMPLE_IMAGE`` environment
    variable.  When the variable is not provided the default image from the
    ``workspace`` directory is used.  A clear ``RuntimeError`` is raised when the
    image is missing to avoid starting a swarm with an invalid configuration.
    """

    image_override = os.getenv("LOCUST_SAMPLE_IMAGE")
    if image_override:
        path = Path(image_override).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Configured sample image '{path}' does not exist or is not a file.")
        return path

    default_path = Path(__file__).resolve().parents[2] / "workspace" / "img_12.jpg"
    if not default_path.is_file():
        raise RuntimeError(
            "Default sample image 'workspace/img_12.jpg' is missing. "
            "Provide a valid path via the LOCUST_SAMPLE_IMAGE environment variable."
        )
    return default_path


def _prepare_payload(image_path: Path) -> tuple[str, bytes, str]:
    """Load the inference payload from disk."""

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return image_path.name, image_path.read_bytes(), mime_type


class OCRUser(HttpUser):
    """Simulated OCR API client issuing inference requests."""

    wait_time = between(0.5, 2.0)
    _sample_payload: ClassVar[tuple[str, bytes, str] | None] = None

    def on_start(self) -> None:
        if OCRUser._sample_payload is None:
            image_path = _resolve_sample_image()
            OCRUser._sample_payload = _prepare_payload(image_path)
        self._file_name, self._payload, self._mime_type = OCRUser._sample_payload

    @task(5)
    def infer(self) -> None:
        files = {"file": (self._file_name, self._payload, self._mime_type)}
        with self.client.post("/infer", files=files, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError as exc:  # pragma: no cover - exercised under load
                response.failure(f"Response is not valid JSON: {exc}")
                return

            if "results" not in payload:
                response.failure("JSON payload did not contain 'results'.")
                return

            response.success()

    @task(1)
    def healthcheck(self) -> None:
        with self.client.get("/healthz", name="healthz", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed with status {response.status_code}")
            else:
                response.success()
