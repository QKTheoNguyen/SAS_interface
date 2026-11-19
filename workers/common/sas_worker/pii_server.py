"""PII service (worker ``pii``): ASR + PII detection, then masking of the anonymized output.

Runs in its own environment (workers/pii): the PII models need transformers 5.x while the
metrics worker's SER model needs 4.x.

- ``GET  /pii/options``    ASR models and PII detectors discovered from quick-convert
- ``POST /pii/obfuscate``  offline: detect on the source, mask the output (section 8)
- ``WS   /pii/stream``     live: detect on the stream input, mask the held-back output

Discovery: concrete classes listed in ``quick_convert.systems.asr.__all__`` (an ``ASRSystem``
implementing ``transcribe_words``) and ``quick_convert.systems.pii.__all__`` (a ``PIIDetector``),
whose constructor needs no arguments. A model is live-capable when it implements
``open_stream()``. Contract: local/PII_requirement.md.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .audio import read_wav, write_wav
from .obfuscate import MASKS, MAX_PADDING_MS, obfuscate
from .pii import LivePII, clean_segments, clean_words

log = logging.getLogger("sas_worker.pii")

SAMPLE_RATE = 16000  # used when an ASR model does not declare `sr`
LABELS = {
    "MoonshineASR": "Moonshine (+ MMS forced alignment)",
    "OpenAIPrivacyFilter": "OpenAI Privacy Filter",
}


@dataclass
class PIIModel:
    id: str
    kind: str  # "asr" | "pii"
    cls: type | None
    status: str = "pending"  # pending -> loading -> ready | unavailable
    error: str | None = None
    obj: Any = field(default=None, repr=False)
    streaming: bool = False  # implements open_stream() (live PII obfuscation)

    @property
    def label(self) -> str:
        return LABELS.get(self.id, self.id)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "error": self.error,
            "default": False,
            "streaming": self.streaming,
        }


def missing_dependency(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return f"Missing package {exc.name!r}: add its quick-convert extra to workers/pii/pyproject.toml"
    return f"{type(exc).__name__}: {exc}"


def no_required_args(cls: type) -> bool:
    params = inspect.signature(cls).parameters.values()
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


def _is_stub(fn: Any) -> bool:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "raise NotImplementedError" in src


def overrides(cls: type, method: str) -> bool:
    """``cls`` has a usable ``method``: not missing, abstract, or a ``raise NotImplementedError`` stub."""
    impl = getattr(cls, method, None)
    return callable(impl) and not getattr(impl, "__isabstractmethod__", False) and not _is_stub(impl)


def discover_package(package: Any, kind: str, accept: Any) -> list[PIIModel]:
    """Concrete classes in ``package.__all__`` that ``accept`` and need no constructor arguments."""
    found = []
    for name in getattr(package, "__all__", []):
        try:
            cls = getattr(package, name)
        except Exception as exc:  # lazily imported class whose dependency is missing
            found.append(PIIModel(name, kind, None, "unavailable", missing_dependency(exc)))
            continue
        if inspect.isclass(cls) and not inspect.isabstract(cls) and no_required_args(cls) and accept(cls):
            found.append(PIIModel(name, kind, cls, streaming=overrides(cls, "open_stream")))
    return found


def discover() -> list[PIIModel]:
    """ASR models with word timestamps and PII detectors. A missing package just lists nothing."""
    found = []
    try:
        from quick_convert.systems import asr
        from quick_convert.systems.asr import ASRSystem

        found += discover_package(
            asr, "asr", lambda cls: issubclass(cls, ASRSystem) and overrides(cls, "transcribe_words")
        )
    except Exception as exc:
        log.warning("ASR discovery skipped: %s", missing_dependency(exc))
    try:
        from quick_convert.systems import pii
        from quick_convert.systems.pii import PIIDetector

        found += discover_package(pii, "pii", lambda cls: issubclass(cls, PIIDetector))
    except Exception as exc:
        log.warning("PII discovery skipped: %s", missing_dependency(exc))
    return found


def instantiate(cls: type, device: torch.device) -> Any:
    kwargs = {"device": str(device)} if "device" in inspect.signature(cls).parameters else {}
    return cls(**kwargs)


def load(model: PIIModel, device: torch.device) -> None:
    """Instantiate and run once, which also triggers checkpoint downloads (lazy-loading models)."""
    model.status = "loading"
    try:
        model.obj = instantiate(model.cls, device)
        if model.kind == "asr":
            noise = torch.randn(1, int(getattr(model.obj, "sr", SAMPLE_RATE))) * 0.01
            model.obj.transcribe_words(noise)
            # Noise transcribes to nothing, which skips word alignment: load the aligner explicitly
            # (MoonshineASR's MMS aligner downloads ~1.2 GB on first use).
            aligner = getattr(model.obj, "aligner", None)
            if callable(getattr(aligner, "align", None)):
                aligner.align(noise.reshape(-1), ["hello"])
        else:
            # Non-empty words so detectors that load weights on first use do it now, not on the first request.
            model.obj.detect([{"word": "hello", "start_s": 0.0, "end_s": 0.5}])
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


def create_pii_app(device: torch.device) -> FastAPI:
    models = {m.id: m for m in discover()}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-model")

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    def load_all() -> None:
        for model in models.values():
            if model.cls is not None:
                load(model, device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        executor.submit(load_all)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="sas_worker:pii", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"worker": "pii", "device": str(device), "models": {}}

    @app.get("/pii/options")
    def pii_options() -> dict:
        return {
            "asr_models": [m.describe() for m in models.values() if m.kind == "asr"],
            "pii_models": [m.describe() for m in models.values() if m.kind == "pii"],
            "masks": list(MASKS),
            "max_padding_ms": MAX_PADDING_MS,
        }

    def ready(model_id: str, kind: str) -> PIIModel:
        model = models.get(model_id)
        if model is None or model.kind != kind:
            raise HTTPException(404, f"Unknown {kind} model {model_id!r}")
        if model.status != "ready":
            raise HTTPException(503, f"{model.label} is {model.status}" + (f": {model.error}" if model.error else ""))
        return model

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
        asr_sr = int(getattr(asr.obj, "sr", SAMPLE_RATE))
        try:
            src = read_wav(await source.read(), asr_sr)
            out_bytes = await output.read()
            out_sr = sf.info(io.BytesIO(out_bytes)).samplerate
            out = read_wav(out_bytes, out_sr)
        except Exception as exc:
            raise HTTPException(400, f"Bad input: {exc}") from exc
        src_seconds = src.size(-1) / asr_sr

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
                    await ws.send_json({"type": "error", "detail": f"{m.label} has no streaming mode (open_stream)"})
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
