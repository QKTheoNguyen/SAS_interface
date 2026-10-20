from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def read_wav(data: bytes, target_sample_rate: int) -> torch.Tensor:
    """Decode WAV/FLAC bytes into a mono ``[1, T]`` float32 tensor at ``target_sample_rate``."""
    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
    if sample_rate != target_sample_rate:
        waveform = AF.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def write_wav(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform as 16-bit PCM WAV, peak-limited to avoid clipping."""
    samples = waveform.detach().float().cpu().reshape(-1).numpy()
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
