"""Metrics service: compares the source and the output of an anonymization.

- Speaker similarity: cosine similarity between speaker embeddings (lower = more private).
- Emotion distance (MED): Euclidean distance between SER emotion vectors, as in
  quick_convert/pipelines/evaluation/metrics/emo/med.py (lower = emotion better preserved).
- F0 contours of both signals, for the panel's overlay plot.
- Naturalness MOS of both signals (UTMOS, not in quick-convert: loaded via torch.hub).
- Spectrograms of both signals, for the panel's heatmaps.
- PII obfuscation (section 8 "Post processing"): ASR + PII detection on the source,
  then the detected segments are masked in the output (see obfuscate.py).

The encoders, SER models and F0 trackers are discovered from quick-convert, so a new
``SpeakerEncoder`` exported by ``quick_convert.components.encoders.speaker``,
a new ``SERSystem`` listed in ``quick_convert.systems.ser.__all__``, or a new feature
extractor producing ``ProsodyMeasurements`` columns shows up in
the UI without changes here (once its quick-convert extra is installed in
workers/metrics/pyproject.toml). The same goes for classes listed in
``quick_convert.systems.asr.__all__`` (an ``ASRSystem`` implementing ``transcribe_words``)
and ``quick_convert.systems.pii.__all__`` (a ``PIIDetector``); contract in local/PII_requirement.md.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import io
import json
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .audio import read_wav, write_wav
from .obfuscate import MASKS, MAX_PADDING_MS, obfuscate
from .pii import LivePII, clean_segments, clean_words

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
    kind: str  # "speaker" | "ser" | "f0" | "mos" | "asr" | "pii"
    cls: type | None
    status: str = "pending"  # pending -> loading -> ready | unavailable
    error: str | None = None
    obj: Any = field(default=None, repr=False)
    streaming: bool = False  # asr / pii: implements open_stream() (live PII obfuscation)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": LABELS.get(self.id, self.id),
            "status": self.status,
            "error": self.error,
            "default": DEFAULTS.get(self.kind) == self.id,
            "streaming": self.streaming,
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
    return found + discover_pii()


def discover_pii() -> list[MetricModel]:
    """ASR models with word timestamps (``quick_convert.systems.asr``) and PII detectors
    (``quick_convert.systems.pii``). A package that does not exist yet just lists nothing."""
    found = []
    try:
        from quick_convert.systems import asr
        from quick_convert.systems.asr import ASRSystem

        base_words = getattr(ASRSystem, "transcribe_words", None)
        found += discover_package(
            asr, "asr", lambda cls: issubclass(cls, ASRSystem) and callable(getattr(cls, "transcribe_words", None))
            and cls.transcribe_words is not base_words
        )
    except Exception as exc:
        log.warning("ASR discovery skipped: %s", missing_dependency(exc))
    try:
        from quick_convert.systems import pii
        from quick_convert.systems.pii import PIIDetector

        found += discover_package(pii, "pii", lambda cls: issubclass(cls, PIIDetector))
    except Exception as exc:
        log.info("PII discovery skipped: %s", missing_dependency(exc))
    return found


def discover_package(package: Any, kind: str, accept: Any) -> list[MetricModel]:
    """Concrete classes in ``package.__all__`` that ``accept`` and need no constructor arguments."""
    found = []
    for name in getattr(package, "__all__", []):
        try:
            cls = getattr(package, name)
        except Exception as exc:  # lazily imported class whose dependency is missing
            found.append(MetricModel(name, kind, None, "unavailable", missing_dependency(exc)))
            continue
        if inspect.isclass(cls) and not inspect.isabstract(cls) and no_required_args(cls) and accept(cls):
            found.append(MetricModel(name, kind, cls, streaming=overrides(cls, "open_stream")))
    return found


def overrides(cls: type, method: str) -> bool:
    """``cls`` has a usable ``method``: not missing, abstract, or a ``raise NotImplementedError`` stub."""
    impl = getattr(cls, method, None)
    return callable(impl) and not getattr(impl, "__isabstractmethod__", False) and not _is_stub(impl)


def _is_stub(fn: Any) -> bool:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "raise NotImplementedError" in src


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
        elif model.kind == "asr":
            model.obj.transcribe_words(torch.randn(1, getattr(model.obj, "sr", SAMPLE_RATE)) * 0.01)
        elif model.kind == "pii":
            model.obj.detect([])
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
        for model in sorted(models.values(), key=lambda m: DEFAULTS.get(m.kind) != m.id):
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

    @app.get("/pii/options")
    def pii_options() -> dict:
        return {
            "asr_models": [m.describe() for m in models.values() if m.kind == "asr"],
            "pii_models": [m.describe() for m in models.values() if m.kind == "pii"],
            "masks": list(MASKS),
            "max_padding_ms": MAX_PADDING_MS,
        }

    @app.post("/pii/obfuscate")
    async def pii_obfuscate(
        source: UploadFile = File(...),
        output: UploadFile = File(...),
        asr_model: str = Form(...),
        pii_model: str = Form(...),
        mask: str = Form("beep"),
        padding_ms: float = Form(100.0),
        offset_s: float = Form(0.0),
    ) -> dict:
        """ASR + PII detection on the clean source; mask the segments in the output (same timing assumed).

        ``offset_s``: how much later the output is than the source (a stream recording is
        shifted by the model's algorithmic delay); 0 for non-streaming conversions."""
        asr = ready(asr_model, "asr")
        pii = ready(pii_model, "pii")
        if mask not in MASKS:
            raise HTTPException(400, f"Unknown mask {mask!r}; expected one of {list(MASKS)}")
        if not 0 <= padding_ms <= MAX_PADDING_MS:
            raise HTTPException(400, f"padding_ms must be in [0, {MAX_PADDING_MS:g}]")
        try:
            src = read_wav(await source.read(), getattr(asr.obj, "sr", SAMPLE_RATE))
            out_bytes = await output.read()
            out_sr = sf.info(io.BytesIO(out_bytes)).samplerate
            out = read_wav(out_bytes, out_sr)
        except Exception as exc:
            raise HTTPException(400, f"Bad input: {exc}") from exc
        src_seconds = src.size(-1) / getattr(asr.obj, "sr", SAMPLE_RATE)

        def work() -> dict:
            start = time.perf_counter()
            words = clean_words(asr.obj.transcribe_words(src), src_seconds)
            segments = clean_segments(pii.obj.detect(words))
            masked = obfuscate(out.numpy(), out_sr, segments, mask=mask, padding_ms=padding_ms, start_s=-offset_s)
            return {
                "words": words,
                "segments": segments,
                "audio_b64": base64.b64encode(write_wav(torch.from_numpy(masked), out_sr)).decode("ascii"),
                "sample_rate": out_sr,
                "source_seconds": src_seconds,
                "output_seconds": out.size(-1) / out_sr,
                "offset_s": offset_s,
                "seconds": time.perf_counter() - start,
            }

        try:
            return await run(work)
        except Exception as exc:
            log.error("PII obfuscation failed\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

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
    ) -> dict:
        encoder = ready(speaker_encoder, "speaker") if speaker_encoder else None
        ser = ready(ser_model, "ser") if ser_model else None
        tracker = ready(f0_tracker, "f0") if f0_tracker else None
        mos = ready(mos_model, "mos") if mos_model else None
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

    live = {"busy": False}

    @app.websocket("/pii/stream")
    async def pii_stream(ws: WebSocket) -> None:
        """Live PII obfuscation, driven by the gateway next to a VC stream.

        1. gateway text  {"type": "start", "asr_model", "pii_model", "mask", "padding_ms",
                          "holdback_ms" (null = auto), "input_sample_rate", "output_sample_rate", "model_delay_ms"}
        2. worker text   {"type": "ready", "holdback_ms", "detector_latency_ms"}
        3. gateway binary uint32 kind + float32 PCM; kind 0 = VC input chunk, 1 = raw VC output chunk.
           For each output chunk the worker sends back one binary frame: the same number of
           masked samples, held back by holdback_ms. Input chunks produce text events
           {"type": "pii_words", "words"} / {"type": "pii_segments", "segments"} (source time).
        4. gateway text {"type": "stop"} -> final events, one binary frame with the held-back
           remainder, then {"type": "pii_end", "leaked_ms", "leaked_segments", "holdback_ms"}.
        Errors: worker text {"type": "error", "detail"}, close.
        """
        await ws.accept()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-stream")
        session: LivePII | None = None
        owner = False

        async def call(fn, *args):
            return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)

        try:
            start = await ws.receive_json()
            try:
                asr, pii = ready(start.get("asr_model", ""), "asr"), ready(start.get("pii_model", ""), "pii")
            except HTTPException as exc:
                await ws.send_json({"type": "error", "detail": exc.detail})
                return
            for m in (asr, pii):
                if not m.streaming:
                    await ws.send_json({"type": "error", "detail": f"{m.id} has no streaming mode (open_stream)"})
                    return
            mask, padding_ms = start.get("mask", "beep"), float(start.get("padding_ms", 100))
            if mask not in MASKS or not 0 <= padding_ms <= MAX_PADDING_MS:
                await ws.send_json({"type": "error", "detail": "Bad mask or padding_ms"})
                return
            if live["busy"]:
                await ws.send_json({"type": "error", "detail": "Live PII obfuscation is busy with another stream."})
                return
            live["busy"] = owner = True
            holdback = start.get("holdback_ms")
            session = await call(
                LivePII,
                asr.obj,
                pii.obj,
                int(start["input_sample_rate"]),
                int(start["output_sample_rate"]),
                float(start.get("model_delay_ms", 0)) / 1000,
                mask,
                padding_ms,
                None if holdback is None else float(holdback),
            )
            await ws.send_json({
                "type": "ready",
                "holdback_ms": session.holdback_s * 1000,
                "detector_latency_ms": session.detector_latency_s * 1000,
            })
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    kind = int.from_bytes(data[:4], "little")
                    pcm = np.frombuffer(data, dtype="<f4", offset=4)
                    if kind == 0:
                        for event in await call(session.push_input, pcm):
                            await ws.send_json(event)
                    else:
                        await ws.send_bytes((await call(session.push_output, pcm)).astype("<f4").tobytes())
                elif json.loads(message.get("text") or "{}").get("type") == "stop":
                    events, tail = await call(session.flush)
                    for event in events:
                        await ws.send_json(event)
                    await ws.send_bytes(tail.astype("<f4").tobytes())
                    await ws.send_json(session.summary())
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.error("live PII failed\n%s", traceback.format_exc())
            try:
                await ws.send_json({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            if session is not None:
                try:
                    await call(session.close)
                except Exception:
                    log.debug(traceback.format_exc())
            pool.shutdown(wait=False)
            if owner:
                live["busy"] = False
            try:
                await ws.close()
            except Exception:
                pass

    return app
