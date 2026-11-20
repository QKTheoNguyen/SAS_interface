"""Metrics service: compares the source and the output of an anonymization.

- Speaker similarity: cosine similarity between speaker embeddings (lower = more private).
- Emotion distance (MED): Euclidean distance between SER emotion vectors, as in
  quick_convert/pipelines/evaluation/metrics/emo/med.py (lower = emotion better preserved).
- F0 contours of both signals, for the panel's overlay plot.
- Naturalness MOS of both signals (UTMOS, not in quick-convert: loaded via torch.hub).
- Spectrograms of both signals, for the panel's heatmaps.
- Speaker similarity over time: per-window cosine from quick-convert's
  ``RollingSpeakerSimilarity`` (quick_convert/components/metrics/streaming_speaker.py),
  kept separate from the full-recording score above.

The encoders, SER models and F0 trackers are discovered from quick-convert, so a new
``SpeakerEncoder`` exported by ``quick_convert.components.encoders.speaker``,
a new ``SERSystem`` listed in ``quick_convert.systems.ser.__all__``, or a new feature
extractor producing ``ProsodyMeasurements`` columns shows up in
the UI without changes here (once its quick-convert extra is installed in
workers/metrics/pyproject.toml).
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import logging
import math
import pkgutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .audio import read_wav

log = logging.getLogger("sas_worker.metrics")

SAMPLE_RATE = 16000  # all quick-convert speaker encoders and the Odyssey SER expect 16 kHz
LABELS = {
    "CosyVoiceCAMPPlusSpeakerEncoder": "CAMPPlus (CosyVoice 3)",
    "ESPnetSpeakerEncoder": "WavLM-ECAPA (ESPnet, VoxCeleb)",
    "PyannoteWeSpeakerEncoder": "WeSpeaker ResNet34 (pyannote)",
    "OdysseySER": "Odyssey WavLM categorical SER",
    "OpenSmileProsodyExtractor": "OpenSMILE ComParE F0",
    "WorldDioF0Extractor": "WORLD DIO (MPM settings)",
    "UTMOS22Strong": "UTMOS22 strong (SpeechMOS)",
}
DEFAULTS = {
    "speaker": "CosyVoiceCAMPPlusSpeakerEncoder",
    "ser": "OdysseySER",
    "f0": "OpenSmileProsodyExtractor",
    "mos": "UTMOS22Strong",
}

# Spectrogram view, as Praat's defaults: 0-5 kHz, 70 dB below the loudest bin.
SPEC_N_FFT = 512
SPEC_WIN = 400  # 25 ms
SPEC_HOP = 160  # 10 ms
SPEC_FMAX = 5000
SPEC_RANGE_DB = 70
SPEC_MAX_FRAMES = 1500  # longer clips are average-pooled in time to keep the payload small

# Speaker similarity over time. The rolling evaluator needs a constant-offset alignment
# (output sample 0 = source time -offset_s). Outputs whose length fits neither "same length as
# the source" (stream recordings: the delayed tail is cut) nor "source + delay" by more than
# this tolerance are treated as unaligned and every window is reported unavailable.
ROLLING_ALIGN_TOLERANCE_S = 0.1
ROLLING_ALIGN_TOLERANCE_REL = 0.02
# CAMPPlus randomly crops inputs longer than 10 s (see quick-convert's
# scripts/evaluate_streaming_speaker_similarity.py), so longer windows are refused.
ROLLING_MAX_WINDOW_S = {"CosyVoiceCAMPPlusSpeakerEncoder": 10.0}


class WorldDioF0Extractor:
    """Pitch track of the Masked Prosody Model (masked_prosody_model.measures.PitchMeasure,
    https://github.com/MiniXC/masked_prosody_model): WORLD DIO + StoneMask at 22.05 kHz,
    hop 256. Mirrored on pyworld because the package pins numpy<2, unlike this env.

    Returns the same ``ProsodyMeasurements`` columns as quick-convert's prosody
    extractors (RMS left at zero) so both go through ``f0_contour``.
    """

    SAMPLE_RATE = 22050
    HOP_LENGTH = 256

    def extract_batch(self, batch):
        import pyworld as pw

        outputs = []
        for waveform, length, sample_rate in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True):
            audio = AF.resample(waveform[: int(length)].float(), int(sample_rate), self.SAMPLE_RATE)
            audio = audio.numpy().astype(np.float64)
            f0, t = pw.dio(audio, self.SAMPLE_RATE, frame_period=self.HOP_LENGTH / self.SAMPLE_RATE * 1000)
            f0 = pw.stonemask(audio, f0, t, self.SAMPLE_RATE)
            f0[f0 < 1e-6] = 0
            voiced = (f0 > 0).astype(np.float64)
            outputs.append(torch.as_tensor(np.stack((t, f0, voiced, np.zeros_like(f0), voiced), axis=-1)).float())
        return outputs


class UTMOS22Strong:
    """UTMOS22 strong learner (Saeki et al., Interspeech 2022): predicted naturalness MOS (1-5)
    of 16 kHz speech. Not in quick-convert; loaded from https://github.com/tarepan/SpeechMOS
    via torch.hub (checkpoint cached under $TORCH_HOME/hub).
    """

    HUB_REPO = "tarepan/SpeechMOS:v1.2.0"

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.hub.load(self.HUB_REPO, "utmos22_strong", trust_repo=True).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, wav: torch.Tensor) -> float:
        return float(self.model(wav.reshape(1, -1).to(self.device), SAMPLE_RATE)[0])


@dataclass
class MetricModel:
    id: str
    kind: str  # "speaker" | "ser" | "f0" | "mos"
    cls: type | None
    status: str = "pending"  # pending -> loading -> ready | unavailable
    error: str | None = None
    obj: Any = field(default=None, repr=False)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": LABELS.get(self.id, self.id),
            "status": self.status,
            "error": self.error,
            "default": DEFAULTS[self.kind] == self.id,
        }


def discover() -> list[MetricModel]:
    from quick_convert.components.encoders import speaker as speaker_encoders
    from quick_convert.components.encoders.speaker import SpeakerEncoder
    from quick_convert.systems import ser
    from quick_convert.systems.ser import SERSystem

    found = []
    for kind, module, base in (("speaker", speaker_encoders, SpeakerEncoder), ("ser", ser, SERSystem)):
        for name in module.__all__:
            try:
                cls = getattr(module, name)
            except Exception as exc:  # lazily imported class whose dependency is missing
                found.append(MetricModel(name, kind, None, "unavailable", missing_dependency(exc)))
                continue
            if inspect.isclass(cls) and issubclass(cls, base) and not inspect.isabstract(cls) and no_required_args(cls):
                found.append(MetricModel(name, kind, cls))
    found += [MetricModel(cls.__name__, "f0", cls) for cls in f0_extractors()]
    found.append(MetricModel(UTMOS22Strong.__name__, "mos", UTMOS22Strong))
    return found


def f0_extractors() -> list[type]:
    """quick-convert feature extractors whose rows are ``ProsodyMeasurements`` (time, F0, voiced, ...)."""
    from quick_convert.components import features
    from quick_convert.components.features.prosody import ProsodyMeasurements
    from quick_convert.components.resources.base import BaseResourceExtractor

    found = []
    for info in pkgutil.walk_packages(features.__path__, f"{features.__name__}."):
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:  # module needing a missing extra; it may not even be an F0 extractor
            log.debug("skipping feature extractor module %s: %s", info.name, exc)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                cls.__module__ == module.__name__
                and issubclass(cls, BaseResourceExtractor)
                and not inspect.isabstract(cls)
                and getattr(cls, "COLUMNS", None) == ProsodyMeasurements.COLUMNS
            ):
                found.append(cls)
    return found + [WorldDioF0Extractor]


def no_required_args(cls: type) -> bool:
    """Skip classes that need a checkpoint path or similar (e.g. SpeechBrainECAPASpeakerEncoder)."""
    params = inspect.signature(cls).parameters.values()
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


def missing_dependency(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return f"Missing package {exc.name!r}: add its quick-convert extra to workers/metrics/pyproject.toml"
    return f"{type(exc).__name__}: {exc}"


def instantiate(cls: type, device: torch.device) -> Any:
    kwargs = {"device": str(device)} if "device" in inspect.signature(cls).parameters else {}
    obj = cls(**kwargs)
    if hasattr(obj, "smile"):
        obj.smile = _DedupSmile(obj.smile)
    return obj


class _DedupSmile:
    """opensmile 2.6 repeats the last LLD frame (same start/end), which quick-convert's
    ProsodyMeasurements rejects as non-increasing timestamps; keep the first copy."""

    def __init__(self, smile: Any):
        self._smile = smile

    def process_signal(self, *args, **kwargs):
        frame = self._smile.process_signal(*args, **kwargs)
        return frame[~frame.index.duplicated()]


def speaker_embedding(encoder: Any, wav: torch.Tensor) -> torch.Tensor:
    return encoder.encode(wav, encoder.sample_rate).values.float().reshape(-1).cpu()


def validate_rolling(encoder_id: str, window_s: float, hop_s: float, offset_s: float) -> None:
    """Raise ValueError with a user-facing message for an unusable window/hop/offset."""
    if not all(math.isfinite(v) for v in (window_s, hop_s, offset_s)):
        raise ValueError("Window, hop and offset must be finite numbers.")
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("Window and hop must be greater than 0 s.")
    if hop_s > window_s:
        raise ValueError("Hop must not be longer than the window.")
    for value in (window_s, hop_s):
        if (Fraction(str(value)) * SAMPLE_RATE).denominator != 1:
            raise ValueError(f"{value} s is not a whole number of samples at {SAMPLE_RATE} Hz.")
    limit = ROLLING_MAX_WINDOW_S.get(encoder_id)
    if limit is not None and window_s > limit:
        raise ValueError(f"{LABELS.get(encoder_id, encoder_id)} supports windows up to {limit:g} s.")


def time_aligned(source_s: float, output_s: float, offset_s: float) -> bool:
    """True when the output length is consistent with a constant offset of ``offset_s``:
    either the whole delayed source is there (source + offset) or the delayed tail was cut
    at the source length (stream recordings)."""
    tolerance = max(ROLLING_ALIGN_TOLERANCE_S, ROLLING_ALIGN_TOLERANCE_REL * source_s)
    return abs(output_s - source_s) <= tolerance or abs(output_s - (source_s + offset_s)) <= tolerance


def rolling_speaker_similarity(
    encoder: Any,
    encoder_id: str,
    src: torch.Tensor,
    out: torch.Tensor,
    window_s: float,
    hop_s: float,
    offset_s: float = 0.0,
) -> dict:
    """Speaker similarity over time with quick-convert's ``RollingSpeakerSimilarity``.

    ``src`` and ``out`` are ``[1, T]`` at SAMPLE_RATE; the output is ``offset_s`` later than the
    source (model delay of a stream recording, 0 otherwise). Runs synchronously: call it from the
    model thread, which owns the encoder. Windows that cannot be scored come back with
    ``speaker_similarity: None`` and a status of ``unavailable`` or ``error``, never a made-up value.
    """
    from quick_convert.components.metrics.streaming_speaker import RollingSpeakerSimilarity

    validate_rolling(encoder_id, window_s, hop_s, offset_s)
    src, out = src.reshape(-1), out.reshape(-1)
    source_s, output_s = src.numel() / SAMPLE_RATE, out.numel() / SAMPLE_RATE
    aligned = time_aligned(source_s, output_s, offset_s)
    start = time.perf_counter()
    core = RollingSpeakerSimilarity(
        encoder,
        SAMPLE_RATE,
        SAMPLE_RATE,
        session_id="post-conversion",
        window_seconds=window_s,
        hop_seconds=hop_s,
        aligned=aligned,
        output_start_seconds=-offset_s,
        # Whole recordings are added at once: the buffers must hold them entirely,
        # or the evaluator keeps only the tail and reports the early windows unavailable.
        max_buffer_seconds=max(source_s, output_s, window_s) + 1.0,
    )
    try:
        core.add_original(src)
        core.add_anonymized(out)
        core.finish()
        windows = [r.to_dict() for r in core.poll()]
    finally:
        core.close()
    ok = [w for w in windows if w["status"] == "ok"]
    if source_s < window_s:
        note = f"Recording is {source_s:.1f} s, shorter than the {window_s:g} s window: no window to score."
    elif not aligned:
        note = (
            f"Alignment unverified: the output is {output_s:.2f} s for a {source_s:.2f} s source"
            f"{f' with a {offset_s * 1000:.0f} ms delay' if offset_s else ''}. This model may not keep "
            "the source timing, so no window was scored."
        )
    else:
        note = None
    return {
        "model": LABELS.get(encoder_id, encoder_id),
        "window_s": window_s,
        "hop_s": hop_s,
        "offset_s": offset_s,
        "aligned": aligned,
        "note": note,
        "source_seconds": source_s,
        "output_seconds": output_s,
        "windows": windows,
        "counts": {
            "ok": len(ok),
            "unavailable": sum(w["status"] == "unavailable" for w in windows),
            "error": sum(w["status"] == "error" for w in windows),
        },
        "mean_compute_ms": sum(w["compute_time_ms"] for w in ok) / len(ok) if ok else None,
        "seconds": time.perf_counter() - start,
    }


def emotion_vector(ser: Any, wav: torch.Tensor) -> np.ndarray:
    """SERSystem.get_labels takes samples with a ``path`` (quick-convert dataset contract)."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, wav.reshape(-1).numpy(), SAMPLE_RATE)
        (vector,) = ser.get_labels([SimpleNamespace(path=tmp.name)])
    return np.asarray(vector, dtype=np.float64).reshape(-1)


def f0_contour(extractor: Any, wav: torch.Tensor) -> dict:
    """Frame times (s) and F0 (Hz, None where unvoiced) from a ProsodyMeasurements-style extractor."""
    wav = wav.reshape(1, -1)
    batch = SimpleNamespace(waveforms=wav, lengths=torch.tensor([wav.shape[-1]]), sample_rates=torch.tensor([SAMPLE_RATE]))
    (values,) = extractor.extract_batch(batch)
    times, f0, voiced = values[:, 0].tolist(), values[:, 1].tolist(), values[:, 2].tolist()
    return {
        "t": [round(t, 4) for t in times],
        "hz": [round(hz, 2) if v else None for hz, v in zip(f0, voiced, strict=True)],
    }


def log_spectrogram(wav: torch.Tensor) -> dict:
    """Log-magnitude STFT up to SPEC_FMAX, quantized to bytes for the canvas heatmap
    (255 = loudest bin, 0 = SPEC_RANGE_DB or more below it), frame-major and base64-encoded."""
    window = torch.hann_window(SPEC_WIN)
    stft = torch.stft(wav.reshape(-1).float(), SPEC_N_FFT, SPEC_HOP, SPEC_WIN, window, center=True, return_complex=True)
    bins = SPEC_FMAX * SPEC_N_FFT // SAMPLE_RATE + 1
    power = stft[:bins].abs().square()
    pool = -(-power.shape[1] // SPEC_MAX_FRAMES)
    if pool > 1:
        power = F.avg_pool1d(power[None], pool, ceil_mode=True)[0]
    db = 10 * torch.log10(power.clamp_min(1e-10))
    level = ((db - db.max()) / SPEC_RANGE_DB + 1).clamp(0, 1) * 255
    data = level.round().to(torch.uint8).T.contiguous().numpy()
    return {
        "frames": data.shape[0],
        "bins": data.shape[1],
        "hop_s": pool * SPEC_HOP / SAMPLE_RATE,
        "fmax": (bins - 1) * SAMPLE_RATE / SPEC_N_FFT,
        "range_db": SPEC_RANGE_DB,
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


def emotion_label(ser: Any, vector: np.ndarray) -> str | None:
    """Top class name when the SER exposes Hugging Face style id2label (Odyssey does)."""
    id2label = getattr(getattr(getattr(ser, "_model", None), "config", None), "id2label", None)
    if not id2label or len(vector) != len(id2label):
        return None
    return str(id2label.get(int(vector.argmax()), id2label.get(str(int(vector.argmax())))))


def load(model: MetricModel, device: torch.device) -> None:
    """Instantiate and run once on noise, which also triggers checkpoint downloads."""
    model.status = "loading"
    try:
        model.obj = instantiate(model.cls, device)
        if hasattr(model.obj, "eval"):
            model.obj.eval()
        noise = torch.randn(1, SAMPLE_RATE) * 0.01
        if model.kind == "speaker":
            speaker_embedding(model.obj, noise)
        elif model.kind == "f0":
            f0_contour(model.obj, noise)
        elif model.kind == "mos":
            model.obj.predict(noise)
        else:
            emotion_vector(model.obj, noise)
        model.status = "ready"
        log.info("%s ready", model.id)
    except Exception as exc:
        model.status, model.error, model.obj = "unavailable", missing_dependency(exc), None
        log.warning("%s unavailable: %s", model.id, model.error)
        log.debug(traceback.format_exc())


def prefetch() -> None:
    for model in discover():
        if model.cls is not None:
            load(model, torch.device("cpu"))


def create_metrics_app(device: torch.device) -> FastAPI:
    models = {m.id: m for m in discover()}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics-model")

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    def load_all() -> None:
        # Defaults first, so the panel is usable as soon as possible.
        for model in sorted(models.values(), key=lambda m: DEFAULTS[m.kind] != m.id):
            if model.cls is not None:
                load(model, device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        executor.submit(load_all)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="sas_worker:metrics", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"worker": "metrics", "device": str(device), "models": {}}

    @app.get("/metrics/options")
    def options() -> dict:
        return {
            "speaker_encoders": [m.describe() for m in models.values() if m.kind == "speaker"],
            "ser_models": [m.describe() for m in models.values() if m.kind == "ser"],
            "f0_trackers": [m.describe() for m in models.values() if m.kind == "f0"],
            "mos_models": [m.describe() for m in models.values() if m.kind == "mos"],
        }

    def ready(model_id: str, kind: str) -> MetricModel:
        model = models.get(model_id)
        if model is None or model.kind != kind:
            raise HTTPException(404, f"Unknown {kind} model {model_id!r}")
        if model.status != "ready":
            raise HTTPException(503, f"{LABELS.get(model_id, model_id)} is {model.status}" + (f": {model.error}" if model.error else ""))
        return model

    @app.post("/metrics/compute")
    async def compute(
        source: UploadFile = File(...),
        output: UploadFile = File(...),
        speaker_encoder: str = Form(""),
        ser_model: str = Form(""),
        f0_tracker: str = Form(""),
        mos_model: str = Form(""),
        spectrogram: bool = Form(False),
        rolling: bool = Form(False),
        rolling_window_s: float = Form(4.0),
        rolling_hop_s: float = Form(2.0),
        offset_s: float = Form(0.0),
    ) -> dict:
        encoder = ready(speaker_encoder, "speaker") if speaker_encoder else None
        ser = ready(ser_model, "ser") if ser_model else None
        tracker = ready(f0_tracker, "f0") if f0_tracker else None
        mos = ready(mos_model, "mos") if mos_model else None
        if rolling:
            if encoder is None:
                raise HTTPException(400, "Speaker similarity over time needs a speaker encoder.")
            try:
                validate_rolling(encoder.id, rolling_window_s, rolling_hop_s, offset_s)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        try:
            src = read_wav(await source.read(), SAMPLE_RATE)
            out = read_wav(await output.read(), SAMPLE_RATE)
        except Exception as exc:
            raise HTTPException(400, f"Bad input: {exc}") from exc

        def work() -> dict:
            start = time.perf_counter()
            result: dict[str, Any] = {}
            if encoder:
                a, b = speaker_embedding(encoder.obj, src), speaker_embedding(encoder.obj, out)
                result["speaker_similarity"] = {
                    "value": float(F.cosine_similarity(a, b, dim=-1)),
                    "model": LABELS.get(encoder.id, encoder.id),
                }
            if encoder and rolling:
                # Separate from the full-recording score above, never averaged into it.
                result["speaker_similarity_over_time"] = rolling_speaker_similarity(
                    encoder.obj, encoder.id, src, out, rolling_window_s, rolling_hop_s, offset_s
                )
            if ser:
                a, b = emotion_vector(ser.obj, src), emotion_vector(ser.obj, out)
                result["emotion_distance"] = {
                    "value": float(np.linalg.norm(a - b)),
                    "model": LABELS.get(ser.id, ser.id),
                    "source_emotion": emotion_label(ser.obj, a),
                    "output_emotion": emotion_label(ser.obj, b),
                }
            if tracker:
                result["f0"] = {
                    "model": LABELS.get(tracker.id, tracker.id),
                    "source": f0_contour(tracker.obj, src),
                    "output": f0_contour(tracker.obj, out),
                }
            if mos:
                result["mos"] = {
                    "model": LABELS.get(mos.id, mos.id),
                    "source": mos.obj.predict(src),
                    "output": mos.obj.predict(out),
                }
            if spectrogram:
                result["spectrogram"] = {"source": log_spectrogram(src), "output": log_spectrogram(out)}
            result["seconds"] = time.perf_counter() - start
            return result

        try:
            return await run(work)
        except Exception as exc:
            log.error("metrics failed\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return app
