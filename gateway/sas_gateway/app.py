from __future__ import annotations

import asyncio
import io
import json
import struct
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from .config import Config, load_config, load_manifests

STATIC_DIR = Path(__file__).parent / "static"


class RevalidatedStaticFiles(StaticFiles):
    """No-build frontend: make browsers revalidate (cheap 304) so an updated app.js is never stale."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response
METRICS_WORKER = "metrics"  # serves /metrics/* instead of models (workers/common/sas_worker/metrics.py)
PII_WORKER = "pii"  # serves /pii/* (workers/common/sas_worker/pii_server.py), own env: workers/pii
# Conversions of a 60 s clip on a slow GPU (or CPU) can take minutes.
CONVERT_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


def create_app(config: Config | None = None, on_ready: Callable[[], None] | None = None) -> FastAPI:
    config = config or load_config()
    manifests = load_manifests(config)
    client = httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if on_ready:
            on_ready()
        yield
        await client.aclose()

    app = FastAPI(title="SAS_interface", lifespan=lifespan)

    async def worker_health(worker: str) -> dict[str, Any] | None:
        try:
            r = await client.get(f"{config.workers[worker]}/health", timeout=2.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            return None

    def manifest_or_404(name: str) -> dict[str, Any]:
        if name not in manifests:
            raise HTTPException(404, f"Unknown model {name!r}")
        return manifests[name]

    async def forward(response: httpx.Response) -> Response:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise HTTPException(response.status_code, detail)
        return Response(
            response.content,
            media_type=response.headers.get("content-type"),
            headers={k: v for k, v in response.headers.items() if k.lower().startswith("x-")},
        )

    @app.get("/api/config")
    def get_config() -> dict:
        return {
            "max_input_seconds": config.max_input_seconds,
            "max_stream_seconds": config.max_stream_seconds,
            "metrics": METRICS_WORKER in config.workers,
            "pii": PII_WORKER in config.workers,
        }

    @app.get("/api/models")
    async def list_models() -> list[dict]:
        healths = dict(zip(config.workers, await asyncio.gather(*map(worker_health, config.workers))))
        out = []
        for manifest in manifests.values():
            health = healths.get(manifest["worker"])
            state = (health or {}).get("models", {}).get(manifest["name"])
            if health is None:
                status, error = "offline", f"Worker {manifest['worker']!r} is not reachable"
            elif state is None:
                status, error = "error", f"Worker {manifest['worker']!r} does not serve this model"
            else:
                status, error = state["status"], state.get("error")
            out.append({
                **manifest,
                "status": status,
                "error": error,
                "warning": (state or {}).get("warning"),
                "device": (health or {}).get("device"),
            })
        return out

    @app.get("/api/models/{name}/choices/{param}")
    async def choices(name: str, param: str) -> Response:
        manifest = manifest_or_404(name)
        try:
            r = await client.get(f"{config.workers[manifest['worker']]}/models/{name}/choices/{param}", timeout=10.0)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Worker unreachable: {exc}") from exc
        return await forward(r)

    def speaker_pool(name: str) -> tuple[Path, dict[str, Any]]:
        """Pool folder and parsed speakers.json of a model (built by scripts/build_reference_pool.py)."""
        pool = manifest_or_404(name).get("speaker_pool")
        if not pool:
            raise HTTPException(404, f"Model {name!r} has no speaker pool")
        folder = config.models_dir / pool
        try:
            return folder, json.loads((folder / "speakers.json").read_text())
        except FileNotFoundError as exc:
            raise HTTPException(
                404, f"No speakers.json in {folder}. Build the pool with scripts/build_reference_pool.py."
            ) from exc

    @app.get("/api/models/{name}/speakers")
    def list_speakers(name: str) -> dict:
        return speaker_pool(name)[1]

    @app.get("/api/models/{name}/speakers/{speaker}.flac")
    def speaker_audio(name: str, speaker: str) -> FileResponse:
        folder, meta = speaker_pool(name)
        # Only ids listed in speakers.json, so the path cannot leave the pool folder.
        if speaker not in {s["id"] for s in meta["speakers"]}:
            raise HTTPException(404, f"Unknown pseudo-speaker {speaker!r}")
        return FileResponse(folder / f"{speaker}.flac", media_type="audio/flac")

    def check_duration(data: bytes, label: str) -> None:
        try:
            info = sf.info(io.BytesIO(data))
        except RuntimeError as exc:
            raise HTTPException(400, f"{label}: unreadable audio ({exc}). Send WAV or FLAC.") from exc
        if info.duration > config.max_input_seconds:
            raise HTTPException(
                413, f"{label} is {info.duration:.1f} s; the limit is {config.max_input_seconds:.0f} s (config.yaml)."
            )

    @app.post("/api/models/{name}/convert")
    async def convert(
        name: str,
        task: str = Form(...),
        params: str = Form("{}"),
        source: UploadFile = File(...),
        targets: list[UploadFile] = File(default=[]),
    ) -> Response:
        manifest = manifest_or_404(name)
        if task not in manifest["tasks"]:
            raise HTTPException(400, f"Model {name!r} does not support task {task!r}")
        if "non-streaming" not in manifest["modes"]:
            raise HTTPException(400, f"Model {name!r} has no non-streaming mode")
        if manifest["tasks"][task].get("target_voice", "none") == "required" and not targets:
            raise HTTPException(400, "This model needs at least one target voice clip")
        json.loads(params)  # reject malformed JSON here rather than in the worker

        src = await source.read()
        check_duration(src, "Source")
        files = [("source", ("source.wav", src, "audio/wav"))]
        for i, t in enumerate(targets):
            data = await t.read()
            check_duration(data, f"Target clip {i + 1}")
            files.append(("targets", (f"target{i}.wav", data, "audio/wav")))

        try:
            r = await client.post(
                f"{config.workers[manifest['worker']]}/models/{name}/convert",
                data={"task": task, "params": params},
                files=files,
                timeout=CONVERT_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Worker unreachable: {exc}") from exc
        return await forward(r)

    def metrics_url() -> str:
        if METRICS_WORKER not in config.workers:
            raise HTTPException(404, "No metrics worker configured (add `metrics` under `workers` in config.yaml)")
        return config.workers[METRICS_WORKER]

    @app.get("/api/metrics/options")
    async def metrics_options() -> Response:
        try:
            r = await client.get(f"{metrics_url()}/metrics/options", timeout=5.0)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Metrics worker unreachable: {exc}") from exc
        return await forward(r)

    def pii_url() -> str:
        if PII_WORKER not in config.workers:
            raise HTTPException(404, "No PII worker configured (add `pii` under `workers` in config.yaml)")
        return config.workers[PII_WORKER]

    @app.get("/api/pii/options")
    async def pii_options() -> Response:
        try:
            r = await client.get(f"{pii_url()}/pii/options", timeout=5.0)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"PII worker unreachable: {exc}") from exc
        return await forward(r)

    @app.post("/api/pii/obfuscate")
    async def pii_obfuscate(
        source: UploadFile = File(...),
        output: UploadFile = File(...),
        asr_model: str = Form(...),
        pii_model: str = Form(...),
        mask: str = Form("beep"),
        padding_ms: float = Form(100.0),
        offset_s: float = Form(0.0),
    ) -> Response:
        url = pii_url()
        files = [
            ("source", ("source.wav", await source.read(), "audio/wav")),
            ("output", ("output.wav", await output.read(), "audio/wav")),
        ]
        data = {
            "asr_model": asr_model,
            "pii_model": pii_model,
            "mask": mask,
            "padding_ms": str(padding_ms),
            "offset_s": str(offset_s),
        }
        try:
            r = await client.post(f"{url}/pii/obfuscate", data=data, files=files, timeout=CONVERT_TIMEOUT)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"PII worker unreachable: {exc}") from exc
        return await forward(r)

    @app.post("/api/metrics/compute")
    async def metrics_compute(
        source: UploadFile = File(...),
        output: UploadFile = File(...),
        speaker_encoder: str = Form(""),
        ser_model: str = Form(""),
        f0_tracker: str = Form(""),
        mos_model: str = Form(""),
        spectrogram: bool = Form(False),
    ) -> Response:
        url = metrics_url()
        files = [
            ("source", ("source.wav", await source.read(), "audio/wav")),
            ("output", ("output.wav", await output.read(), "audio/wav")),
        ]
        try:
            r = await client.post(
                f"{url}/metrics/compute",
                data={
                    "speaker_encoder": speaker_encoder,
                    "ser_model": ser_model,
                    "f0_tracker": f0_tracker,
                    "mos_model": mos_model,
                    "spectrogram": str(spectrogram).lower(),
                },
                files=files,
                timeout=CONVERT_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Metrics worker unreachable: {exc}") from exc
        return await forward(r)

    @app.websocket("/api/models/{name}/stream")
    async def stream(ws: WebSocket, name: str) -> None:
        """Relay frames both ways between the browser and the worker (protocol in sas_worker/server.py).

        Live PII obfuscation: when the browser's start message carries ``pii`` options, the
        gateway also opens the PII worker's ``/pii/stream`` (protocol in sas_worker/pii_server.py)
        once the model is ready. Input chunks are copied to it, raw output chunks go through it,
        and each output frame sent to the browser becomes
        ``uint32 seq + float32 ms + uint32 n_raw + float32 raw[n_raw] + float32 masked[...]``.
        The ready message gains ``pii: {holdback_ms, detector_latency_ms}`` or ``pii: {error}``.
        If the PII service fails the stream goes on unmasked (fail open: masked = raw) and the
        browser gets ``{"type": "pii_error", "detail"}``.
        """
        await ws.accept()
        manifest = manifests.get(name)
        if manifest is None or "streaming" not in manifest["modes"]:
            await ws.send_json({"type": "error", "detail": f"Model {name!r} does not support streaming"})
            await ws.close()
            return
        url = config.workers[manifest["worker"]].replace("http", "ws", 1) + f"/models/{name}/stream"
        pii_opts: dict[str, Any] | None = None
        pii: dict[str, Any] = {"ws": None, "reader": None, "pending": deque(), "last_header": b"\0" * 8}

        def pii_frame(header: bytes, raw: bytes, masked: bytes) -> bytes:
            return header + struct.pack("<I", len(raw) // 4) + raw + masked

        async def pii_fail(detail: str) -> None:
            """Fail open: drop the PII session, release frames still waiting for their masked version."""
            conn, pii["ws"] = pii["ws"], None
            if conn is None:
                return
            await ws.send_json({"type": "pii_error", "detail": detail})
            while pii["pending"]:
                header, raw = pii["pending"].popleft()
                await ws.send_bytes(pii_frame(header, raw, raw))
            try:
                await conn.close()
            except Exception:
                pass

        async def pii_reader(conn) -> None:
            try:
                async for message in conn:
                    if isinstance(message, bytes):
                        if pii["pending"]:
                            header, raw = pii["pending"].popleft()
                        else:  # held-back remainder sent at stop
                            header, raw = pii["last_header"][:4] + struct.pack("<f", 0.0), b""
                        await ws.send_bytes(pii_frame(header, raw, message))
                        continue
                    event = json.loads(message)
                    if event.get("type") == "error":
                        raise RuntimeError(event.get("detail"))
                    await ws.send_text(message)
                    if event.get("type") == "pii_end":
                        return
                raise RuntimeError("PII service closed the connection")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await pii_fail(f"Live PII obfuscation stopped, output is no longer masked: {exc}")

        async def pii_open(ready: dict[str, Any]) -> dict[str, Any]:
            """Start the PII session for a ready VC stream; return what to tell the browser."""
            if PII_WORKER not in config.workers:
                return {"error": "No PII worker configured: live PII obfuscation is off."}
            stream_url = config.workers[PII_WORKER].replace("http", "ws", 1) + "/pii/stream"
            try:
                conn = await ws_connect(stream_url, max_size=None, ping_interval=None)
                await conn.send(json.dumps({
                    "type": "start",
                    **{k: pii_opts.get(k) for k in ("asr_model", "pii_model", "mask", "padding_ms", "holdback_ms")},
                    "input_sample_rate": ready["sample_rate"],
                    "output_sample_rate": ready["output_sample_rate"],
                    "model_delay_ms": ready.get("algorithmic_latency_ms", 0),
                }))
                reply = json.loads(await asyncio.wait_for(conn.recv(), timeout=60))
            except (OSError, WebSocketException, asyncio.TimeoutError, ValueError) as exc:
                return {"error": f"Live PII obfuscation unavailable: {exc or type(exc).__name__}"}
            if reply.get("type") != "ready":
                await conn.close()
                return {"error": f"Live PII obfuscation unavailable: {reply.get('detail', reply)}"}
            pii["ws"] = conn
            pii["reader"] = asyncio.create_task(pii_reader(conn))
            return {"holdback_ms": reply["holdback_ms"], "detector_latency_ms": reply["detector_latency_ms"]}

        try:
            async with ws_connect(url, max_size=None, ping_interval=None) as upstream:
                streaming = False  # True once the worker is ready: binary frames are audio chunks

                async def browser_to_worker() -> None:
                    nonlocal pii_opts
                    while True:
                        message = await ws.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        data = message.get("bytes")
                        if data is None:
                            text = message["text"]
                            if pii_opts is None and not streaming:
                                start = json.loads(text)
                                if start.get("type") == "start" and start.get("pii"):
                                    pii_opts = start.pop("pii")
                                    text = json.dumps(start)
                            await upstream.send(text)
                            continue
                        await upstream.send(data)
                        if streaming and pii["ws"] is not None:
                            try:
                                await pii["ws"].send(struct.pack("<I", 0) + data[4:])
                            except (OSError, WebSocketException) as exc:
                                await pii_fail(f"Live PII obfuscation stopped, output is no longer masked: {exc}")

                async def worker_to_browser() -> None:
                    nonlocal streaming
                    async for message in upstream:
                        if isinstance(message, bytes):
                            if pii_opts is None:
                                await ws.send_bytes(message)
                                continue
                            header, raw = message[:8], message[8:]
                            pii["last_header"] = header
                            if pii["ws"] is None:  # requested but unavailable or failed: fail open
                                await ws.send_bytes(pii_frame(header, raw, raw))
                                continue
                            pii["pending"].append((header, raw))
                            try:
                                await pii["ws"].send(struct.pack("<I", 1) + raw)
                            except (OSError, WebSocketException) as exc:
                                await pii_fail(f"Live PII obfuscation stopped, output is no longer masked: {exc}")
                            continue
                        event = json.loads(message)
                        if event.get("type") == "ready":
                            streaming = True
                            if pii_opts is not None:
                                event["pii"] = await pii_open(event)
                                message = json.dumps(event)
                        elif event.get("type") == "end" and pii["ws"] is not None:
                            # Flush the held-back output before telling the browser the stream ended.
                            try:
                                await pii["ws"].send(json.dumps({"type": "stop"}))
                                await asyncio.wait_for(asyncio.shield(pii["reader"]), timeout=30)
                            except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
                                await pii_fail(f"Live PII flush failed: {exc or 'timeout'}")
                        await ws.send_text(message)

                tasks = [asyncio.create_task(browser_to_worker()), asyncio.create_task(worker_to_browser())]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in tasks:
                    task.cancel()
        except (OSError, WebSocketException) as exc:
            try:
                await ws.send_json({"type": "error", "detail": f"Worker unreachable: {exc}"})
            except Exception:
                pass
        finally:
            if pii["reader"] is not None:
                pii["reader"].cancel()
            if pii["ws"] is not None:
                try:
                    await pii["ws"].close()
                except Exception:
                    pass
            try:
                await ws.close()
            except Exception:
                pass

    app.mount("/", RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
