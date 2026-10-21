# SPDX-License-Identifier: GPL-3.0-or-later
# Imports GPL-3.0 code from third_party/Voice-Privacy-Challenge-2026, so this file is GPL too.
from __future__ import annotations

import multiprocessing
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sas_worker import ModelAdapter

VPC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Voice-Privacy-Challenge-2026"
COEFF_RANGE = (0.5, 0.9)  # mc_coeff_min / mc_coeff_max in VPC configs/track1/anon_mcadams.yaml


def enter_vpc_root() -> None:
    """VPC code imports its top-level `utils` package and finds checkpoints under `exp/` relative to its root.
    Every model of the vpc worker expects this, so changing the working directory is safe."""
    os.chdir(VPC_ROOT)
    if str(VPC_ROOT) not in sys.path:
        sys.path.insert(0, str(VPC_ROOT))


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        enter_vpc_root()
        # The module forces the "spawn" start method on import; undo that for the rest of the worker.
        start_method = multiprocessing.get_start_method(allow_none=True)
        from anonymization.modules.mcadams.anonymise_dir_mcadams_rand_seed import anonym_v2

        multiprocessing.set_start_method(start_method, force=True)
        self.anonym = anonym_v2

    def prefetch(self) -> None:
        pass  # no checkpoints

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        if params.get("random_coefficient", True):
            coefficient = random.uniform(*COEFF_RANGE)
        else:
            coefficient = float(params.get("coefficient", 0.8))
        samples = self.anonym(
            freq=self.sample_rate,
            samples=source.reshape(-1).double().numpy(),
            winLengthinms=int(params.get("window_ms", 20)),
            shiftLengthinms=int(params.get("shift_ms", 10)),
            lp_order=int(params.get("lp_order", 20)),
            mcadams=coefficient,
        )
        # VPC writes the result peak-normalized to full scale.
        samples = samples / (np.max(np.abs(samples)) + 1e-9) * 0.99
        return torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
