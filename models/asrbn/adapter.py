from __future__ import annotations

import random
from typing import Any

import torch

from sas_worker import ModelAdapter


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        from quick_convert.systems.anonymization import ASRBNAnonymizer

        self.system = ASRBNAnonymizer()
        # ASRBNAnonymizer picks its own device; move it to the one the worker chose.
        self.system.device = device
        self.system.model.to(device)
        self.speakers = [str(s) for s in getattr(self.system.model, "spk", [])]

    def choices(self, param: str) -> list[str]:
        if param != "target_speaker":
            raise KeyError(param)
        return self.speakers

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        target = str(params.get("target_speaker") or "random")
        if target == "random":
            if not self.speakers:
                raise ValueError("This checkpoint exposes no speaker list; pick a target speaker ID explicitly.")
            target = random.choice(self.speakers)
        elif self.speakers and target not in self.speakers:
            raise ValueError(f"Unknown target speaker {target!r}")
        return self.system.anonymize(source, sample_rate=self.sample_rate, target_speaker=target)
