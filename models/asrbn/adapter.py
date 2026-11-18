from __future__ import annotations

import random
from typing import Any

import torch

from sas_worker import ModelAdapter


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        from quick_convert.systems.conversion import ASRBNConversionSystem

        self.system = ASRBNConversionSystem(device=device)
        if self.system.model is None:
            # quick-convert chains `.to(device).eval()`, but SA-toolkit's Net.eval() returns None.
            model = torch.hub.load(
                "deep-privacy/SA-toolkit",
                "anonymization",
                tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
                trust_repo=True,
            ).to(device)
            model.eval()
            self.system.model = model
        self.speakers = [str(s) for s in getattr(self.system.model, "spk", [])]

    def choices(self, param: str) -> list[str]:
        if param != "target_speaker":
            raise KeyError(param)
        return self.speakers

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        from quick_convert.systems.conversion import SpeakerIdentity

        target = str(params.get("target_speaker") or "random")
        if target == "random":
            if not self.speakers:
                raise ValueError("This checkpoint exposes no speaker list; pick a target speaker ID explicitly.")
            target = random.choice(self.speakers)
        elif self.speakers and target not in self.speakers:
            raise ValueError(f"Unknown target speaker {target!r}")
        prepared = self.system.prepare_target(SpeakerIdentity(target))
        return self.system.convert(source, target=prepared, sample_rate=self.sample_rate)
