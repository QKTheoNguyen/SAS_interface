from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from .adapter import ModelAdapter, StreamSession
from .audio import read_wav, write_wav

log = logging.getLogger("sas_worker")

# Binary stream frames. Client -> worker: uint32 seq + float32 PCM.
# Worker -> client: uint32 seq of the last input chunk consumed + float32 compute ms + float32 PCM.
IN_HEADER = struct.Struct("<I")
OUT_HEADER = struct.Struct("<If")


@dataclass
class ModelSlot:
    adapter: ModelAdapter
    status: str = "pending"  # pending -> loading -> ready | error
    error: str | None = None
    streaming: bool = False


def create_app(worker: str, adapters: list[ModelAdapter], device: torch.device) -> FastAPI:
    slots = {a.name: ModelSlot(a) for a in adapters}
    # Every model call runs on this one thread: GPU work never overlaps, and
    # compiled CUDA graphs are always replayed from the thread that recorded them.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{worker}-model")

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    def load_all() -> None:
        for slot in slots.values():
            slot.status = "loading"
            log.info("Loading %s on %s", slot.adapter.name, device)
            try:
                slot.adapter.device = device
                slot.adapter.load(device)
                slot.status = "ready"
                log.info("%s ready", slot.adapter.name)
            except Exception as exc:
                slot.status, slot.error = "error", f"{type(exc).__name__}: {exc}"
                log.error("Failed to load %s\n%s", slot.adapter.name, traceback.format_exc())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        executor.submit(load_all)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title=f"sas_worker:{worker}", lifespan=lifespan)

    def ready_slot(name: str) -> ModelSlot:
        slot = slots.get(name)
        if slot is None:
            raise HTTPException(404, f"Model {name!r} is not served by worker {worker!r}")
        if slot.status != "ready":
            raise HTTPException(503, f"Model {name!r} is {slot.status}" + (f": {slot.error}" if slot.error else ""))
        if slot.streaming:
            raise HTTPException(503, f"Model {name!r} is busy with a live stream; try again when it stops.")
        return slot

    @app.get("/health")
    def health() -> dict:
        return {
            "worker": worker,
            "device": str(device),
            "models": {
                n: {"status": s.status, "error": s.error, "warning": getattr(s.adapter, "warning", None), "streaming": s.streaming}
                for n, s in slots.items()
            },
        }

    @app.get("/models/{name}/choices/{param}")
    def choices(name: str, param: str) -> dict:
        slot = slots.get(name)
        if slot is None or slot.status != "ready":
            raise HTTPException(503, f"Model {name!r} is not ready")
        try:
            return {"choices": slot.adapter.choices(param)}
        except KeyError:
            raise HTTPException(404, f"Model {name!r} has no dynamic choices for {param!r}") from None

    @app.post("/models/{name}/convert")
    async def convert(
        name: str,
        task: str = Form(...),
        params: str = Form("{}"),
        source: UploadFile = File(...),
        targets: list[UploadFile] = File(default=[]),
    ) -> Response:
        adapter = ready_slot(name).adapter
        try:
            parsed = json.loads(params)
            src = read_wav(await source.read(), adapter.sample_rate)
            tgts = [read_wav(await t.read(), adapter.sample_rate) for t in targets]
        except Exception as exc:
            raise HTTPException(400, f"Bad input: {exc}") from exc

        def timed():
            start = time.perf_counter()
            out = adapter.convert(task, src, tgts, parsed)
            return out, time.perf_counter() - start

        try:
            out, compute = await run(timed)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.error("convert failed for %s\n%s", name, traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

        return Response(
            write_wav(out, adapter.output_sample_rate),
            media_type="audio/wav",
            headers={"X-Compute-Seconds": f"{compute:.3f}"},
        )

    @app.websocket("/models/{name}/stream")
    async def stream(ws: WebSocket, name: str) -> None:
        """Protocol:
        1. client text  {"type": "start", "task", "params", "n_targets"}
        2. client binary x n_targets: WAV/FLAC files
        3. worker text  {"type": "ready", "sample_rate", "output_sample_rate", "chunk_samples", "algorithmic_latency_ms"}
        4. client binary IN frames <-> worker binary OUT frames
        5. client text  {"type": "stop"} -> worker text {"type": "end"}, close
        Errors: worker text {"type": "error", "detail"}, close.
        """
        await ws.accept()
        session: StreamSession | None = None
        slot: ModelSlot | None = None  # set only once this connection owns the model
        try:
            try:
                adapter = ready_slot(name).adapter
            except HTTPException as exc:
                await ws.send_json({"type": "error", "detail": exc.detail})
                return
            start = await ws.receive_json()
            targets = [read_wav(await ws.receive_bytes(), adapter.sample_rate) for _ in range(int(start.get("n_targets", 0)))]
            if slots[name].streaming:
                await ws.send_json({"type": "error", "detail": f"Model {name!r} is busy with another live stream."})
                return
            slot = slots[name]
            slot.streaming = True
            try:
                session = await run(adapter.open_stream, start["task"], targets, start.get("params") or {})
            except (ValueError, NotImplementedError) as exc:
                await ws.send_json({"type": "error", "detail": str(exc)})
                return
            await ws.send_json({
                "type": "ready",
                "sample_rate": session.input_sample_rate,
                "output_sample_rate": session.output_sample_rate,
                "chunk_samples": session.chunk_samples,
                "algorithmic_latency_ms": session.algorithmic_latency_ms,
            })

            def process(chunk: np.ndarray):
                t0 = time.perf_counter()
                out = session.process(torch.from_numpy(chunk)[None])
                return out.detach().float().reshape(-1).cpu().numpy(), (time.perf_counter() - t0) * 1000

            n = session.chunk_samples
            pending = np.zeros(0, dtype=np.float32)
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    (seq,) = IN_HEADER.unpack_from(data)
                    pending = np.concatenate([pending, np.frombuffer(data, dtype="<f4", offset=IN_HEADER.size)])
                    while len(pending) >= n:
                        chunk, pending = pending[:n].copy(), pending[n:]
                        out, ms = await run(process, chunk)
                        await ws.send_bytes(OUT_HEADER.pack(seq, ms) + out.astype("<f4").tobytes())
                elif json.loads(message.get("text") or "{}").get("type") == "stop":
                    await ws.send_json({"type": "end"})
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.error("stream failed for %s\n%s", name, traceback.format_exc())
            try:
                await ws.send_json({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            if session is not None:
                await run(session.close)
            if slot is not None:
                slot.streaming = False
            try:
                await ws.close()
            except Exception:
                pass

    return app
