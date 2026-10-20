from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import soxr
import torch


def read_wav(data: bytes, target_sample_rate: int) -> torch.Tensor:
    """Decode WAV/FLAC bytes into a mono ``[1, T]`` float32 tensor at ``target_sample_rate``."""
    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate != target_sample_rate:
        # soxr: ~100x faster than torchaudio's sinc resampler on CPU for ratios like 16k -> 44.1k.
        mono = soxr.resample(mono, sample_rate, target_sample_rate, quality="HQ")
    return torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32)).unsqueeze(0)


def write_wav(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform as 16-bit PCM WAV, peak-limited to avoid clipping."""
    samples = waveform.detach().float().cpu().reshape(-1).numpy()
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
