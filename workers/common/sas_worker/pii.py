"""PII detection glue shared by the offline endpoint and the live session.

Contract of the quick-convert modules: local/PII_requirement.md.

Live timeline (all times in seconds from stream start):
- source time: the VC input = what ASR / PII see. Words and segments are in source time.
- output time: the raw VC output. A segment at source time t sits at t + model_delay_s.
- the listener gets the output held back by ``holdback_s``, so a segment returned up to
  ``holdback_s`` after its audio *started* is still masked before it is sent. That is why the
  sessions' ``max_decision_latency_s`` is measured from the start of a word / segment.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import soxr
import torch

from .obfuscate import obfuscate
from .pii_streaming import open_asr_session, open_pii_session


def clean_words(words: list[dict], duration_s: float | None = None) -> list[dict]:
    """Validate words; keep the optional live tags ``line`` (revisable line id) and ``final``."""
    out = []
    for w in words:
        start, end = max(0.0, float(w["start_s"])), float(w["end_s"])
        if duration_s is not None:
            end = min(duration_s, end)
        word = {"word": str(w["word"]), "start_s": start, "end_s": end}
        if w.get("line") is not None:
            word["line"], word["final"] = int(w["line"]), bool(w.get("final", False))
        out.append(word)
    return out


def clean_segments(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        start, end = float(s["start_s"]), float(s["end_s"])
        if end <= start:
            continue
        out.append({
            "start_s": start,
            "end_s": end,
            "label": str(s.get("label", "PII")),
            "confidence": float(s.get("confidence", 1.0)),
            "text": str(s.get("text", "")),
        })
    return sorted(out, key=lambda s: s["start_s"])


def uncovered(start: float, end: float, segments: list[dict]) -> float:
    """Length of ``[start, end]`` not covered by any of ``segments``."""
    if end <= start:
        return 0.0
    covered, cursor = 0.0, start
    for s in sorted(segments, key=lambda s: s["start_s"]):
        lo, hi = max(s["start_s"], cursor), min(s["end_s"], end)
        if hi > lo:
            covered += hi - lo
            cursor = hi
    return (end - start) - covered


class LivePII:
    """One live session: streaming ASR -> streaming PII on the input, held-back masking of the output.

    ``push_output`` returns exactly as many samples as it receives, delayed by the holdback
    (the first ``holdback`` samples are silence); ``flush`` returns the held-back remainder.
    ``push_input`` (detection) and ``push_output`` (masking) may run on different threads, so a
    slow ASR pass never delays the audio; the shared segment list is guarded by a lock.

    Words tagged with ``line`` belong to a revisable line: a later batch for the same line
    replaces its earlier words (the UI does the same).
    """

    def __init__(
        self,
        asr: Any,
        pii: Any,
        input_sample_rate: int,
        output_sample_rate: int,
        model_delay_s: float,
        mask: str,
        padding_ms: float,
        holdback_ms: float | None = None,
    ):
        self.asr_session = open_asr_session(asr)
        self.pii_session = open_pii_session(pii)
        self.lock = threading.Lock()
        self.asr_sr = int(getattr(asr, "sr", 16000))
        self.resampler = (
            soxr.ResampleStream(input_sample_rate, self.asr_sr, 1, dtype="float32", quality="HQ")
            if input_sample_rate != self.asr_sr
            else None
        )
        self.out_sr = output_sample_rate
        self.model_delay_s = model_delay_s
        self.mask, self.padding_ms = mask, padding_ms
        self.detector_latency_s = float(self.asr_session.max_decision_latency_s) + float(
            self.pii_session.max_decision_latency_s
        )
        auto = self.detector_latency_s + padding_ms / 1000
        self.holdback_s = auto if holdback_ms is None else max(0.0, holdback_ms / 1000)
        hold = int(round(self.holdback_s * output_sample_rate))
        self.buffer = np.zeros(hold, dtype=np.float32)  # pre-filled silence = the holdback
        self.emitted = -hold  # absolute raw-output index of buffer[0]
        self.segments: list[dict] = []  # output time, for obfuscate()
        self.leaked_s = 0.0
        self.leaked_segments = 0

    def _detect(self, words: list[dict], final: bool = False) -> list[dict]:
        events = []
        words = clean_words(words)
        if words:
            events.append({"type": "pii_words", "words": words})
        segments = list(self.pii_session.push(words)) if words else []
        if final:
            segments += list(self.pii_session.flush())
        segments = clean_segments(segments)
        if segments:
            with self.lock:
                sent_until = self.emitted / self.out_sr  # output time already sent to the listener
                for s in segments:
                    start, end = s["start_s"] + self.model_delay_s, s["end_s"] + self.model_delay_s
                    # Leaked = already sent AND not masked by an earlier segment (padding aside).
                    leak = uncovered(start, min(end, sent_until), self.segments)
                    self.segments.append({"start_s": start, "end_s": end})
                    if leak > 0:
                        self.leaked_s += leak
                        self.leaked_segments += 1
            events.append({"type": "pii_segments", "segments": segments})
        return events

    def push_input(self, chunk: np.ndarray) -> list[dict]:
        """Feed one VC input chunk (input sample rate) to the detectors; return UI events."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.resampler is not None:
            chunk = self.resampler.resample_chunk(chunk)
        if not chunk.size:
            return []
        return self._detect(self.asr_session.push(torch.from_numpy(np.ascontiguousarray(chunk))[None]))

    def push_output(self, chunk: np.ndarray) -> np.ndarray:
        """Feed one raw VC output chunk; return the same number of masked, held-back samples."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.buffer = np.concatenate([self.buffer, chunk])
        n = len(chunk)
        return self._emit(n)

    def _emit(self, n: int) -> np.ndarray:
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        with self.lock:
            segments, start = list(self.segments), self.emitted
            self.emitted += len(out)  # counted as sent now: a segment arriving meanwhile is a leak
        return obfuscate(out, self.out_sr, segments, self.mask, self.padding_ms, start_s=start / self.out_sr)

    def flush(self) -> tuple[list[dict], np.ndarray]:
        """End of stream: final words / segments, then the held-back remainder of the output."""
        tail = self.resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True) if self.resampler else None
        words = list(self.asr_session.push(torch.from_numpy(tail)[None])) if tail is not None and tail.size else []
        words += list(self.asr_session.flush())
        events = self._detect(words, final=True)
        return events, self._emit(len(self.buffer))

    def summary(self) -> dict:
        return {
            "type": "pii_end",
            "leaked_ms": self.leaked_s * 1000,
            "leaked_segments": self.leaked_segments,
            "holdback_ms": self.holdback_s * 1000,
        }

    def close(self) -> None:
        for session in (self.asr_session, self.pii_session):
            close = getattr(session, "close", None)
            if callable(close):
                close()
