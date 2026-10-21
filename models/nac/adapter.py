# SPDX-License-Identifier: GPL-3.0-or-later
# Imports GPL-3.0 code from third_party/Voice-Privacy-Challenge-2026, so this file is GPL too.
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

from sas_worker import ModelAdapter
from sas_worker.download import fetch_zip

VPC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Voice-Privacy-Challenge-2026"
# Paths and URL from VPC configs/track1/anon_nac.yaml and anonymization/pipelines/nac/install.sh.
CHECKPOINT_DIR = VPC_ROOT / "exp" / "nac_models"
MAPPINGS_DIR = VPC_ROOT / "exp" / "nac_mappings"
VOICE_DIR = MAPPINGS_DIR / "suno_voices" / "v2"
MAPPINGS_URL = "https://github.com/m-pana/nac-requirements/releases/download/1.0/nac_speaker_mappings.zip"


def enter_vpc_root() -> None:
    """VPC code imports its top-level `utils` package and finds checkpoints under `exp/` relative to its root.
    Every model of the vpc worker expects this, so changing the working directory is safe."""
    os.chdir(VPC_ROOT)
    if str(VPC_ROOT) not in sys.path:
        sys.path.insert(0, str(VPC_ROOT))


class Adapter(ModelAdapter):
    sample_rate = 24000  # EnCodec / Bark

    def load(self, device: torch.device) -> None:
        enter_vpc_root()
        fetch_zip(MAPPINGS_URL, MAPPINGS_DIR, marker=VOICE_DIR)
        # Coqui keeps the HuBERT tokenizer in ~/.local/share/tts; keep it next to the other checkpoints.
        # Must run before BarkConfig is imported, which reads the directory into a class default.
        import TTS.utils.generic_utils as coqui_utils

        coqui_utils.get_user_data_dir = lambda appname: CHECKPOINT_DIR / "user_data" / appname
        from anonymization.modules.nac.anonymizer import Anonymizer

        self.anonymizer = Anonymizer(checkpoint_dir=str(CHECKPOINT_DIR), voice_dirs=[str(VOICE_DIR)]).to(device)
        self.voices = sorted(p.name for p in VOICE_DIR.iterdir() if p.is_dir())
        if device.type != "cuda":
            self.warning = "Running on CPU: Bark takes minutes per utterance."

    def choices(self, param: str) -> list[str]:
        if param != "target_voice":
            raise KeyError(param)
        return self.voices

    @torch.inference_mode()
    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        voice = str(params.get("target_voice") or "random")
        if voice == "random":
            voice = random.choice(self.voices)
        elif voice == "unprompted":
            voice = "random"  # Bark's name for an empty history prompt
        elif voice not in self.voices:
            raise ValueError(f"Unknown target voice {voice!r}")
        audio = self.anonymizer(
            source.to(self.device),
            target_voice_id=voice,
            coarse_temperature=float(params.get("coarse_temperature", 0.7)),
        )
        return torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
