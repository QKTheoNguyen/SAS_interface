"""Streaming sessions for live PII obfuscation (``LivePII``).

A model can provide its own ``open_stream()`` (contract: local/PII_requirement.md, section 5).
Otherwise the adapters here are used:

- ``MoonshineASR`` -> :class:`MoonshineStreamSession`: Moonshine's streaming transcriber, whose
  current line is rewritten about every 0.5 s and completed at a pause. Each updated line is
  force-aligned to its audio with the model's MMS aligner (Moonshine's own streamed word times
  are unreliable) and returned as words tagged ``line`` / ``final``. A later update of the same
  line replaces its earlier words.
- any ``PIIDetector`` -> :class:`DetectorStreamSession`: quick-convert's ``StreamingPIIMasker``
  on the current line, with the previous lines as left context. Only segments not returned
  before are returned, so segments stay append-only even though lines are revised.

Same approach as quick-convert's scripts/pii_live_demo.py (detect on partial lines, never
withdraw a beep), with the delaying / beeping done by ``LivePII`` + ``obfuscate()``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

# Worst case from the start of a word to its words being returned: the word itself (~0.6 s),
# Moonshine's update interval (0.5 s, grows when passes are slow), the pass and the alignment.
MOONSHINE_LATENCY_S = 2.0
# Detection runs on every line update; this covers the privacy filter's forward pass.
DETECTOR_LATENCY_S = 0.3
MOONSHINE_UPDATE_INTERVAL_S = 0.5
DEDUPE_TOLERANCE_S = 0.05  # re-alignments of the same words move their bounds by a few ms


class MoonshineStreamSession:
    """Streaming ASR session on a loaded ``MoonshineASR`` (input: 16 kHz chunks)."""

    max_decision_latency_s = MOONSHINE_LATENCY_S

    def __init__(self, asr: Any):
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged

        self.sr = int(asr.sr)
        self.aligner = asr.aligner
        self._events = (LineTextChanged, LineCompleted)
        self.stream = asr._get_transcriber().create_stream(update_interval=MOONSHINE_UPDATE_INTERVAL_S)
        self.stream.add_listener(self._on_event)
        self._changed: OrderedDict[int, dict] = OrderedDict()  # line index -> latest snapshot
        # Moonshine line ids are 64-bit: JSON numbers lose precision in the browser, so number lines 0, 1, 2...
        self._line_index: dict[int, int] = {}
        self.audio = np.zeros(0, dtype=np.float32)  # whole stream so far, for alignment
        self.stream.start()

    def _on_event(self, event: Any) -> None:
        # Exceptions raised here are swallowed by moonshine_voice: only record, work in push().
        if isinstance(event, self._events) and event.line is not None:
            line = event.line
            index = self._line_index.setdefault(line.line_id, len(self._line_index))
            self._changed[index] = {
                "text": " ".join(line.text.split()),
                "start_s": float(line.start_time),
                "final": bool(line.is_complete),
            }

    def _lines(self) -> list[dict]:
        words = []
        for line_id, line in self._changed.items():
            words += self._align(line_id, line)
        self._changed.clear()
        return words

    def _align(self, line_id: int, line: dict) -> list[dict]:
        tokens = line["text"].split()
        tag = {"line": line_id, "final": line["final"]}
        if not tokens:
            return [{"word": "", "start_s": line["start_s"], "end_s": line["start_s"], **tag}]  # cleared line
        start = max(0, int(line["start_s"] * self.sr))
        audio = self.audio[start:]
        aligned = self.aligner.align(audio, tokens) if len(audio) else None
        if aligned is None:  # cannot align: one entry covering the line so far, so PII in it is still masked
            end = max(line["start_s"] + 0.01, len(self.audio) / self.sr)
            return [{"word": " ".join(tokens), "start_s": line["start_s"], "end_s": end, **tag}]
        base = start / self.sr
        return [{"word": w["word"], "start_s": base + w["start_s"], "end_s": base + w["end_s"], **tag} for w in aligned]

    def push(self, chunk: torch.Tensor) -> list[dict]:
        samples = chunk.reshape(-1).float().numpy()
        self.audio = np.concatenate([self.audio, samples])
        self.stream.add_audio(samples.tolist(), self.sr)  # may run a transcription pass -> _on_event
        return self._lines()

    def flush(self) -> list[dict]:
        self.stream.stop()  # transcribes the rest and completes the last line
        return self._lines()

    def close(self) -> None:
        self.stream.close()


class DetectorStreamSession:
    """Streaming PII session on any ``PIIDetector`` (uses its ``detect_text``)."""

    max_decision_latency_s = DETECTOR_LATENCY_S

    def __init__(self, detector: Any):
        from quick_convert.systems.pii import StreamingPIIMasker

        self.masker = StreamingPIIMasker(detector)
        self.emitted: list[dict] = []

    def _new(self, segment: dict) -> bool:
        """Not already returned: each revision of a line re-detects the same entity, re-aligned a few
        ms differently. A segment that extends an earlier one (more words) is new."""
        tol = DEDUPE_TOLERANCE_S
        return not any(
            old["label"] == segment["label"]
            and old["start_s"] - tol <= segment["start_s"]
            and segment["end_s"] <= old["end_s"] + tol
            for old in self.emitted
        )

    def _detect(self, words: list[dict], final: bool) -> list[dict]:
        from quick_convert.systems.pii.base import timed_spans

        words = [w for w in words if w["word"]]
        text = " ".join(w["word"] for w in words)
        spans = self.masker.spans(text)
        if final and text:
            self.masker.add_context(text)
        out = [s for s in timed_spans(spans, words) if self._new(s)] if spans else []
        self.emitted += out
        return out

    def push(self, words: list[dict]) -> list[dict]:
        """Words of revised lines (``line`` key: each line's full current words), or plain final words."""
        lines: OrderedDict[Any, list[dict]] = OrderedDict()
        for w in words:
            lines.setdefault(w.get("line"), []).append(w)
        segments = []
        for line_id, line_words in lines.items():
            final = line_id is None or any(w.get("final") for w in line_words)
            segments += self._detect(line_words, final)
        return segments

    def flush(self) -> list[dict]:
        return []


def open_asr_session(asr: Any) -> Any:
    if callable(getattr(asr, "open_stream", None)) and not _stub(asr.open_stream):
        return asr.open_stream()
    if type(asr).__name__ == "MoonshineASR":
        return MoonshineStreamSession(asr)
    raise ValueError(f"{type(asr).__name__} has no streaming mode")


def open_pii_session(pii: Any) -> Any:
    if callable(getattr(pii, "open_stream", None)) and not _stub(pii.open_stream):
        return pii.open_stream()
    if callable(getattr(pii, "detect_text", None)):
        return DetectorStreamSession(pii)
    raise ValueError(f"{type(pii).__name__} has no streaming mode")


def asr_can_stream(cls: type, has_open_stream: bool) -> bool:
    return has_open_stream or cls.__name__ == "MoonshineASR"


def pii_can_stream(cls: type, has_open_stream: bool) -> bool:
    return has_open_stream or callable(getattr(cls, "detect_text", None))


def _stub(fn: Any) -> bool:
    import inspect

    try:
        return "raise NotImplementedError" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False
