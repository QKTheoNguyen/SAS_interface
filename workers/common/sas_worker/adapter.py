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
