# SPDX-License-Identifier: GPL-3.0-or-later
# Imports GPL-3.0 code from third_party/Voice-Privacy-Challenge-2026, so this file is GPL too.
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from sas_worker import ModelAdapter
from sas_worker.download import fetch, fetch_zip

VPC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Voice-Privacy-Challenge-2026"
# From VPC anonymization/pipelines/ssl/install.sh.
MODELS_ZIP_URL = "https://duke.app.box.com/shared/static/qe1idwbhzaueyoziimjm6wo56olhtnf7"
HUBERT_URL = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"


def enter_vpc_root() -> None:
    """VPC code imports its top-level `utils` package and finds checkpoints under `exp/` relative to its root.
    Every model of the vpc worker expects this, so changing the working directory is safe."""
    os.chdir(VPC_ROOT)
    if str(VPC_ROOT) not in sys.path:
        sys.path.insert(0, str(VPC_ROOT))


class Adapter(ModelAdapter):
    sample_rate = 16000

    def _download(self):
        enter_vpc_root()
        from anonymization.modules.ssl import anonymizer

        fetch_zip(MODELS_ZIP_URL, VPC_ROOT / "exp", marker=VPC_ROOT / anonymizer.SOFT_MODEL_PATH)
        fetch(HUBERT_URL, VPC_ROOT / anonymizer.HUBERT_MODEL_PATH)
        return anonymizer

    def prefetch(self) -> None:
        self._download()

    def load(self, device: torch.device) -> None:
        anonymizer = self._download()
        self.pipeline = anonymizer.SelectionBasedAnonymizationPipeline(device=str(device))

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        # The pipeline reuses the first pseudo-speaker for later calls unless its cache is reset.
        self.pipeline.reset_cache()
        audio = self.pipeline.process_utterance(
            source.reshape(-1).numpy(),
            sample_rate=self.sample_rate,
            world_size=int(params.get("world_size", 50)),
            region_size=int(params.get("region_size", 10)),
            flag_proximity=str(params.get("proximity", "random")),
            gender_pool=bool(params.get("gender_pool", False)),
            flag_cross_gender=bool(params.get("cross_gender", False)),
        )
        return torch.from_numpy(audio).unsqueeze(0)
