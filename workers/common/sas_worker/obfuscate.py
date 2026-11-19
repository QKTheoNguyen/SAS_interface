"""PII obfuscation: mask time segments of a waveform with a beep, silence or noise.

Pure numpy, no model. Segments come from a PII detector run on the *source* audio and
are applied unchanged to the converted output (same duration assumed).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

MASKS = ("beep", "silence", "noise")
BEEP_HZ = 1000.0
BEEP_PEAK = 0.25  # about -12 dBFS
FADE_MS = 5.0
MAX_PADDING_MS = 300.0


def merge_segments(
    segments: Iterable[Mapping[str, float]], padding_s: float, duration_s: float
) -> list[tuple[float, float]]:
    """Pad each ``{start_s, end_s}`` segment, clip it to ``[0, duration_s]`` and merge overlaps."""
    spans = []
    for seg in segments:
        start = max(0.0, float(seg["start_s"]) - padding_s)
        end = min(duration_s, float(seg["end_s"]) + padding_s)
        if end > start:
            spans.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def obfuscate(
    wav: np.ndarray,
    sample_rate: int,
    segments: Iterable[Mapping[str, float]],
    mask: str = "beep",
    padding_ms: float = 100.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a copy of mono ``wav`` with every (padded, merged) segment replaced by ``mask``.

    Inside a segment the original audio is fully removed, not mixed under the mask.
    The first and last ``FADE_MS`` of each segment crossfade between audio and mask to avoid clicks.
    """
    if mask not in MASKS:
        raise ValueError(f"Unknown mask {mask!r}; expected one of {MASKS}")
    if not 0 <= padding_ms <= MAX_PADDING_MS:
        raise ValueError(f"padding_ms must be in [0, {MAX_PADDING_MS:g}]")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    out = wav.copy()
    spans = merge_segments(segments, padding_ms / 1000, len(wav) / sample_rate)
    if not spans:
        return out

    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
    fade = max(1, int(FADE_MS / 1000 * sample_rate))
    for start_s, end_s in spans:
        a, b = int(round(start_s * sample_rate)), int(round(end_s * sample_rate))
        n = b - a
        if n <= 0:
            continue
        if mask == "beep":
            fill = BEEP_PEAK * np.sin(2 * np.pi * BEEP_HZ * np.arange(n) / sample_rate)
        elif mask == "noise":
            fill = rng.standard_normal(n) * rms
        else:
            fill = np.zeros(n)
        # Gain of the mask: ramps 0 -> 1 over the first `fade` samples and back at the end.
        k = min(fade, n // 2)
        gain = np.ones(n)
        if k:
            ramp = np.linspace(0.0, 1.0, k, endpoint=False)
            gain[:k], gain[n - k :] = ramp, ramp[::-1]
        out[a:b] = (1 - gain) * wav[a:b] + gain * fill
    return out.astype(np.float32)
