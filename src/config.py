from __future__ import annotations

"""Configuration dataclasses used across the Triton OCR client."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TritonClientConfig:
    """Configuration required to build a Triton client instance.

    Attributes:
        url: Triton inference server URL ("host:port").
        protocol: Communication protocol ("http" or "grpc").
        verbose: Enable verbose logging for Triton client libraries.
        ssl: Enable SSL/TLS connections.
        certificate_chain: Optional path to the certificate chain file.
        timeout: Optional per-request timeout in seconds.
    """

    url: str = "192.168.7.9:8001"
    protocol: str = "http"
    verbose: bool = False
    ssl: bool = False
    certificate_chain: Optional[str] = None
    timeout: Optional[int] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        protocol = self.protocol.lower()
        if protocol not in {"http", "grpc"}:
            raise ValueError(f"Unsupported protocol '{self.protocol}'. Use 'http' or 'grpc'.")
        object.__setattr__(self, "protocol", protocol)


@dataclass(frozen=True)
class ModelIOConfig:
    """Describe an input tensor expected by a Triton model."""

    name: str
    dtype: str
    binary_data: bool = True


@dataclass(frozen=True)
class ModelOutputConfig:
    """Describe an output tensor produced by a Triton model."""

    name: str
    binary_data: bool = True


@dataclass(frozen=True)
class TritonModelConfig:
    """Aggregate configuration for a Triton model invocation."""

    model_name: str
    inputs: List[ModelIOConfig] = field(default_factory=list)
    outputs: List[ModelOutputConfig] = field(default_factory=list)
