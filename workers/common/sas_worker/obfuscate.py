"""PII obfuscation: mask time segments of a waveform with a beep, silence or noise.

Pure numpy, no model. Segments come from a PII detector run on the *source* audio and
are applied unchanged to the converted output (same duration assumed).

The same function serves the offline path (whole clip, ``start_s=0``) and the live path
(one chunk of a longer signal, ``start_s`` = time of the chunk's first sample). The fade
gain and the beep phase are computed in absolute sample time, so a segment split across
chunks is masked exactly as if the whole signal had been processed at once.
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
    segments: Iterable[Mapping[str, float]], padding_s: float, duration_s: float | None = None
) -> list[tuple[float, float]]:
    """Pad each ``{start_s, end_s}`` segment, clip it to ``[0, duration_s]`` and merge overlaps."""
    spans = []
    for seg in segments:
        start = max(0.0, float(seg["start_s"]) - padding_s)
        end = float(seg["end_s"]) + padding_s
        if duration_s is not None:
            end = min(duration_s, end)
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
    start_s: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a copy of mono ``wav`` with every (padded, merged) segment replaced by ``mask``.

    ``start_s`` is the time of ``wav[0]`` on the segments' timeline. Inside a segment the
    original audio is fully removed, not mixed under the mask. The first and last ``FADE_MS``
    of each segment crossfade between audio and mask to avoid clicks.
    """
    if mask not in MASKS:
        raise ValueError(f"Unknown mask {mask!r}; expected one of {MASKS}")
    if not 0 <= padding_ms <= MAX_PADDING_MS:
        raise ValueError(f"padding_ms must be in [0, {MAX_PADDING_MS:g}]")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    out = wav.copy()
    spans = merge_segments(segments, padding_ms / 1000)
    if not spans or not wav.size:
        return out

    first = int(round(start_s * sample_rate))  # absolute index of wav[0]
    rng = np.random.default_rng(seed + max(first, 0))
    rms = float(np.sqrt(np.mean(wav**2)))
    fade = max(1, int(FADE_MS / 1000 * sample_rate))
    for span_start, span_end in spans:
        a, b = int(round(span_start * sample_rate)), int(round(span_end * sample_rate))
        lo, hi = max(a, first), min(b, first + len(wav))  # part of the span inside this wav
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)  # absolute sample indices
        if mask == "beep":
            fill = BEEP_PEAK * np.sin(2 * np.pi * BEEP_HZ * idx / sample_rate)
        elif mask == "noise":
            fill = rng.standard_normal(hi - lo) * rms
        else:
            fill = np.zeros(hi - lo)
        # Mask gain: ramps 0 -> 1 over the span's first k samples and back over its last k.
        k = min(fade, (b - a) // 2)
        gain = np.minimum(1.0, np.minimum(idx - a, b - 1 - idx) / k) if k else np.ones(hi - lo)
        seg = slice(lo - first, hi - first)
        out[seg] = (1 - gain) * wav[seg] + gain * fill
    return out.astype(np.float32)
