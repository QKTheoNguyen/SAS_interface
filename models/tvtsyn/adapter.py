from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import constants as hf_constants

from sas_worker import ModelAdapter, StreamSession
from sas_worker.audio import read_wav

REPO_ROOT = Path(__file__).resolve().parents[2]
# Same LibriSpeech train-clean-360 pseudo-speakers as StreamVoiceAnon anonymization.
REFERENCES_DIR = REPO_ROOT / "models" / "streamvoiceanon" / "references"
HF_REPO = "warisqr007/TVTSyn"
# TVTSyn keeps its SpeechBrain x-vector + ECAPA models in ~/.cache by default; keep them with the HF cache.
SPEAKER_CACHE = Path(hf_constants.HF_HOME) / "tvtsyn-speaker"


class Adapter(ModelAdapter):
    sample_rate = 16000

    def load(self, device: torch.device) -> None:
        from tvtsyn_inference import TVTSyn
        from tvtsyn_inference.speaker import SpeakerEncoder

        self.model = TVTSyn.from_pretrained(HF_REPO, backend="pytorch", device=str(device))
        self.runtime = self.model.runtime
        self.speaker_encoder = SpeakerEncoder(device=str(device), cache_dir=SPEAKER_CACHE,
                                              models=self.model.config["speaker_models"])
        # Conditioning is a fixed [1, 704] embedding, so the pool is embedded once here.
        self.pool = {
            p.stem: self._embed([read_wav(p.read_bytes(), self.sample_rate)])
            for p in sorted(REFERENCES_DIR.glob("*.flac"))
        }
        if device.type != "cuda":
            self.warning = "Running on CPU: streaming may fall behind real time on a busy node."

    def choices(self, param: str) -> list[str]:
        if param != "reference_speaker":
            raise KeyError(param)
        return list(self.pool)

    def _embed(self, clips: list[torch.Tensor]) -> np.ndarray:
        return self.speaker_encoder.encode(torch.cat(clips, dim=-1)[0].numpy(), self.sample_rate)

    def _speaker(self, task: str, targets: list[torch.Tensor], params: dict[str, Any]) -> np.ndarray:
        if targets:
            return self._embed(targets)
        if task == "voice_conversion":
            raise ValueError("Voice conversion needs at least one target voice clip.")
        name = str(params.get("reference_speaker", "random"))
        if name == "random":
            name = random.choice(list(self.pool))
        if name not in self.pool:
            raise ValueError(f"Unknown pseudo-speaker {name!r}")
        return self.pool[name]

    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        audio = self.runtime.whole(source[0].numpy(), self._speaker(task, targets, params))
        return torch.from_numpy(audio)

    def open_stream(self, task: str, targets: list[torch.Tensor], params: dict[str, Any]) -> StreamSession:
        return TVTSynStream(self.runtime, self._speaker(task, targets, params))


class TVTSynStream(StreamSession):
    """Feeds the backend's 100 ms steps directly (tvtsyn_inference's StreamingSession adds buffering we do not need)."""

    input_sample_rate = output_sample_rate = Adapter.sample_rate

    def __init__(self, runtime, speaker: np.ndarray):
        self.runtime = runtime
        self.speaker = speaker
        self.chunk_samples = runtime.chunk_samples
        # Output trails input by one step: 80 ms encoder lookahead, rounded up to the 100 ms step.
        self.algorithmic_latency_ms = self.chunk_samples / self.input_sample_rate * 1000
        self.state = runtime.initial_state()

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        out, self.state, ready = self.runtime.step(chunk[0].numpy(), self.speaker, self.state)
        # Silence until the lookahead is filled keeps output aligned with input, one chunk each.
        return torch.from_numpy(out) if ready else torch.zeros(self.chunk_samples)

    def close(self) -> None:
        self.state = None
