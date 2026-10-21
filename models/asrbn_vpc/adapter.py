from __future__ import annotations

import random
from typing import Any

import torch

from sas_worker import ModelAdapter

# Manifest `variant` -> SA-toolkit tag, as listed in VPC configs/track1/anon_asrbn.yaml.
TAGS = {
    "B5": "hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
    "B6": "hifigan_bn_tdnnf_600h_vq_48_v1",
    "wav2vec2 (no VQ)": "hifigan_bn_tdnnf_wav2vec2_100h_aug_v1",
    "TDNN-F (no VQ)": "hifigan_bn_tdnnf_600h_aug_v1",
}
DEFAULT_VARIANT = "B5"


class _Wav:
    """What SA-toolkit's ``get_f0`` expects (VPC anonymise_dir.py wraps audio the same way)."""

    def __init__(self, wav: torch.Tensor):
        self.wav = wav


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        self.models: dict[str, torch.nn.Module] = {}
        self.speakers = [str(s) for s in self._model(DEFAULT_VARIANT).spk]

    def prefetch(self) -> None:
        self.device = torch.device("cpu")
        self.models = {}
        for variant in TAGS:
            self._model(variant)

    def _model(self, variant: str) -> torch.nn.Module:
        if variant not in TAGS:
            raise ValueError(f"Unknown checkpoint {variant!r}")
        if variant not in self.models:
            model = torch.hub.load(
                "deep-privacy/SA-toolkit",
                "anonymization",
                tag_version=TAGS[variant],
                exit_if_new_version=False,
                trust_repo=True,
            )
            # SA-toolkit's model overrides .to() / .eval() without returning self, so don't chain them.
            model.to(self.device)
            model.eval()
            self.models[variant] = model
        return self.models[variant]

    def choices(self, param: str) -> list[str]:
        if param != "target_speaker":
            raise KeyError(param)
        return self.speakers

    @torch.inference_mode()
    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        model = self._model(str(params.get("variant") or DEFAULT_VARIANT))
        speakers = [str(s) for s in model.spk]
        target = str(params.get("target_speaker") or "random")
        if target == "random":
            target = random.choice(speakers)
        elif target not in speakers:
            raise ValueError(f"Unknown target speaker {target!r}")

        # Per-utterance version of VPC anonymise_dir.process_wav: YAAPT F0 on CPU, then conversion.
        f0 = model.get_f0(_Wav(source)).reshape(1, -1)
        model.set_f0(f0.to(self.device))
        converted = model.convert(source.to(self.device), target=[target]).cpu()
        return converted.reshape(1, -1)[:, : source.shape[-1]]
