from __future__ import annotations

import random
from typing import Any

import torch

from sas_worker import ModelAdapter


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        from quick_convert.systems.conversion import PSST1ConversionSystem
        from quick_convert.systems.ser import SpeechBrainSER

        self.system = PSST1ConversionSystem(emotion_recognizer=SpeechBrainSER(device=device), device=device)

    def prefetch(self) -> None:
        self.load(torch.device("cpu"))
        # The emotion recognizer loads lazily; run it once so its checkpoint is downloaded too.
        self.system.emotion_recognizer.predict(torch.randn(1, self.sample_rate) * 0.01, self.sample_rate)

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        from quick_convert.systems.conversion import GeneratedVoice

        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**63)
        emotion = str(params.get("emotion") or "auto")
        w = float(params.get("w", 0.25))
        steps = int(params.get("steps", 10))
        if not 0 <= w <= 1:
            raise ValueError("w must be in [0, 1].")
        if steps < 1:
            raise ValueError("steps must be a positive integer.")

        # w and steps are system attributes; requests run one at a time per worker.
        self.system.w, self.system.steps = w, steps
        target = self.system.prepare_target(GeneratedVoice(seed=seed))
        return self.system.convert(
            source,
            target=target,
            sample_rate=self.sample_rate,
            emotion=None if emotion == "auto" else emotion,
        )
