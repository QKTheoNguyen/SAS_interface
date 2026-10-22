"""Metrics service: compares the source and the output of an anonymization.

- Speaker similarity: cosine similarity between speaker embeddings (lower = more private).
- Emotion distance (MED): Euclidean distance between SER emotion vectors, as in
  quick_convert/pipelines/evaluation/metrics/emo/med.py (lower = emotion better preserved).
- F0 contours of both signals, for the panel's overlay plot.

The encoders, SER models and F0 trackers are discovered from quick-convert, so a new
``SpeakerEncoder`` exported by ``quick_convert.components.speaker.speaker_encoders``,
a new ``SERSystem`` listed in ``quick_convert.systems.ser.__all__``, or a new feature
extractor producing ``ProsodyMeasurements`` columns shows up in
the UI without changes here (once its quick-convert extra is installed in
workers/metrics/pyproject.toml).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
}
DEFAULTS = {"speaker": "CosyVoiceCAMPPlusSpeakerEncoder", "ser": "OdysseySER", "f0": "OpenSmileProsodyExtractor"}


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


@dataclass
class MetricModel:
    id: str
    kind: str  # "speaker" | "ser" | "f0"
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
    from quick_convert.components.speaker import speaker_encoders
    from quick_convert.components.speaker.speaker_encoders import SpeakerEncoder
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
            if inspect.isclass(cls) and issubclass(cls, base) and not inspect.isabstract(cls):
                found.append(MetricModel(name, kind, cls))
    found += [MetricModel(cls.__name__, "f0", cls) for cls in f0_extractors()]
    return found


def f0_extractors() -> list[type]:
    """quick-convert feature extractors whose rows are ``ProsodyMeasurements`` (time, F0, voiced, ...)."""
    from quick_convert.components import feature_extractors
    from quick_convert.components.feature_extractors.base import BaseFeatureExtractor
    from quick_convert.components.prosody import ProsodyMeasurements

    found = []
    for info in pkgutil.iter_modules(feature_extractors.__path__):
        try:
            module = importlib.import_module(f"{feature_extractors.__name__}.{info.name}")
        except Exception as exc:  # module needing a missing extra; it may not even be an F0 extractor
            log.debug("skipping feature extractor module %s: %s", info.name, exc)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                cls.__module__ == module.__name__
                and issubclass(cls, BaseFeatureExtractor)
                and not inspect.isabstract(cls)
                and getattr(cls, "COLUMNS", None) == ProsodyMeasurements.COLUMNS
            ):
                found.append(cls)
    return found + [WorldDioF0Extractor]


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
    ) -> dict:
        encoder = ready(speaker_encoder, "speaker") if speaker_encoder else None
        ser = ready(ser_model, "ser") if ser_model else None
        tracker = ready(f0_tracker, "f0") if f0_tracker else None
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
            result["seconds"] = time.perf_counter() - start
            return result

        try:
            return await run(work)
        except Exception as exc:
            log.error("metrics failed\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return app
