from __future__ import annotations

from typing import Any

import torch

from sas_worker import ModelAdapter


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        # Same model as quick-convert's KNNVCAnonymizer, loaded directly because that
        # wrapper does not pass a device and torch.hub's knn_vc defaults to CUDA.
        self.knn = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=str(device)
        )

    @torch.inference_mode()
    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        if not targets:
            raise ValueError("kNN-VC needs at least one target voice clip.")
        topk = int(params.get("topk", 4))
        # kNN-VC moves tensors to its device itself; VAD trimming runs on CPU.
        matching_set = self.knn.get_matching_set(targets)
        query = self.knn.get_features(source)
        return self.knn.match(query, matching_set, topk=topk)
