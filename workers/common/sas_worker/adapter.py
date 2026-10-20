from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelAdapter(ABC):
    """Contract between a worker and one model.

    Audio handed to :meth:`convert` is already mono, float32, shaped ``[1, T]``
    and resampled to :attr:`sample_rate`. The returned waveform must use
    :attr:`output_sample_rate`.
    """

    sample_rate: int = 16000
    warning: str | None = None  # shown in the UI, e.g. a degraded mode

    def __init__(self, manifest: dict[str, Any]):
        self.manifest = manifest
        self.name: str = manifest["name"]
        self.device = torch.device("cpu")

    @property
    def output_sample_rate(self) -> int:
        return self.sample_rate

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load weights onto ``device``. Called once, before any request."""

    def prefetch(self) -> None:
        """Download checkpoints without needing a GPU (run by ``setup.sh``).

        The default loads the model on CPU, which triggers every download.
        """
        self.load(torch.device("cpu"))

    def choices(self, param: str) -> list[str]:
        """Values for a manifest parameter declared with ``choices_from: worker``."""
        raise KeyError(param)

    @abstractmethod
    def convert(
        self,
        task: str,
        source: torch.Tensor,
        targets: list[torch.Tensor],
        params: dict[str, Any],
    ) -> torch.Tensor:
        """Run one non-streaming conversion and return a ``[T]`` or ``[1, T]`` waveform."""

    def open_stream(self, task: str, targets: list[torch.Tensor], params: dict[str, Any]) -> StreamSession:
        """Start a streaming session. Only models listing ``streaming`` in their manifest ``modes`` implement this."""
        raise NotImplementedError(f"{self.name} does not support streaming")


class StreamSession(ABC):
    """One live conversion. The worker calls :meth:`process` with consecutive
    ``[1, chunk_samples]`` chunks at :attr:`input_sample_rate`, always from the
    same thread, and :meth:`close` once at the end."""

    input_sample_rate: int
    output_sample_rate: int
    chunk_samples: int
    algorithmic_latency_ms: float  # delay the model adds on top of chunking and transport

    @abstractmethod
    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        """Convert one chunk and return the output samples produced for it."""

    def close(self) -> None:
        pass
