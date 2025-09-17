from __future__ import annotations

"""Base utilities for interacting with the Triton Inference Server."""

import logging
from typing import Dict, Iterable, Mapping

import numpy as np

from src.config import ModelIOConfig, ModelOutputConfig, TritonClientConfig, TritonModelConfig


class TritonClientError(RuntimeError):
    """Raised when the Triton client fails to execute an inference request."""


logger = logging.getLogger(__name__)


class BaseTritonClient:
    """Base class that encapsulates common Triton client behaviour.

    The class is responsible for instantiating the low level Triton client and
    exposes a protected ``_infer`` helper that child classes can leverage to
    perform model specific inferences.
    """

    def __init__(self, client_config: TritonClientConfig) -> None:
        self.client_config = client_config
        self._client_module = self._import_client_module(client_config.protocol)
        self._client = self._create_client()

    def _import_client_module(self, protocol: str):  # pragma: no cover - import wrapper
        if protocol == "grpc":
            import tritonclient.grpc as grpcclient

            return grpcclient
        import tritonclient.http as httpclient

        return httpclient

    def _create_client(self):  # pragma: no cover - thin wrapper around SDK
        client_kwargs = {
            "url": self.client_config.url,
            "verbose": self.client_config.verbose,
        }
        if self.client_config.protocol == "http":
            client_kwargs.update(
                {
                    "ssl": self.client_config.ssl,
                }
            )
        else:
            # gRPC client shares the same keyword names for TLS options.
            client_kwargs.update(
                {
                    "ssl": self.client_config.ssl,
                }
            )
        return self._client_module.InferenceServerClient(**client_kwargs)

    @property
    def client(self):
        return self._client

    def _create_infer_input(self, io_config: ModelIOConfig, data: np.ndarray):
        infer_input = self._client_module.InferInput(io_config.name, data.shape, datatype=io_config.dtype)
        # ``binary_data`` defaults to True which is appropriate for most tensor payloads.
        infer_input.set_data_from_numpy(data)
        return infer_input

    def _create_requested_output(self, output_config: ModelOutputConfig):
        return self._client_module.InferRequestedOutput(
            output_config.name
        )

    def _infer(self, model_config: TritonModelConfig, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        infer_inputs = []
        for io_config in model_config.inputs:
            if io_config.name not in inputs:
                raise KeyError(f"Missing input '{io_config.name}' for model '{model_config.model_name}'.")
            tensor = inputs[io_config.name]
            if not isinstance(tensor, np.ndarray):
                raise TypeError(
                    f"Input '{io_config.name}' for model '{model_config.model_name}' must be a numpy array."
                )
            infer_inputs.append(self._create_infer_input(io_config, tensor))

        requested_outputs = [self._create_requested_output(output) for output in model_config.outputs]

        infer_kwargs = {
            "model_name": model_config.model_name,
            "inputs": infer_inputs,
            "outputs": requested_outputs if requested_outputs else None,
        }
        if self.client_config.timeout is not None:
            infer_kwargs["client_timeout"] = self.client_config.timeout

        logger.debug(
            "Sending Triton inference request for model '%s' with %d input(s) and %d requested output(s).",
            model_config.model_name,
            len(infer_inputs),
            len(requested_outputs) if requested_outputs else 0,
        )

        try:
            response = self.client.infer(**infer_kwargs)
        except Exception as exc:  # pragma: no cover - relies on Triton server interaction
            raise TritonClientError(
                f"Triton inference failed for model '{model_config.model_name}': {exc}"
            ) from exc
        logger.debug("Triton inference for model '%s' completed successfully.", model_config.model_name)
        results = {}
        output_names: Iterable[ModelOutputConfig] = model_config.outputs
        for output in output_names:
            results[output.name] = response.as_numpy(output.name)
        if not output_names:
            logger.debug("Model '%s' requested all outputs by default.", model_config.model_name)
        return results
