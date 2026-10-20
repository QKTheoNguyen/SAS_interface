from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .adapter import ModelAdapter
from .audio import read_wav, write_wav

log = logging.getLogger("sas_worker")


@dataclass
class ModelSlot:
    adapter: ModelAdapter
    status: str = "pending"  # pending -> loading -> ready | error
    error: str | None = None


def create_app(worker: str, adapters: list[ModelAdapter], device: torch.device) -> FastAPI:
    slots = {a.name: ModelSlot(a) for a in adapters}
    # One GPU job at a time per worker: requests queue instead of competing for memory.
    gpu_lock = threading.Lock()

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
        threading.Thread(target=load_all, daemon=True).start()
        yield

    app = FastAPI(title=f"sas_worker:{worker}", lifespan=lifespan)

    def ready_slot(name: str) -> ModelSlot:
        slot = slots.get(name)
        if slot is None:
            raise HTTPException(404, f"Model {name!r} is not served by worker {worker!r}")
        if slot.status != "ready":
            raise HTTPException(503, f"Model {name!r} is {slot.status}" + (f": {slot.error}" if slot.error else ""))
        return slot

    @app.get("/health")
    def health() -> dict:
        return {
            "worker": worker,
            "device": str(device),
            "models": {n: {"status": s.status, "error": s.error} for n, s in slots.items()},
        }

    @app.get("/models/{name}/choices/{param}")
    def choices(name: str, param: str) -> dict:
        slot = ready_slot(name)
        try:
            return {"choices": slot.adapter.choices(param)}
        except KeyError:
            raise HTTPException(404, f"Model {name!r} has no dynamic choices for {param!r}") from None

    @app.post("/models/{name}/convert")
    def convert(
        name: str,
        task: str = Form(...),
        params: str = Form("{}"),
        source: UploadFile = File(...),
        targets: list[UploadFile] = File(default=[]),
    ) -> Response:
        adapter = ready_slot(name).adapter
        try:
            parsed = json.loads(params)
            src = read_wav(source.file.read(), adapter.sample_rate)
            tgts = [read_wav(t.file.read(), adapter.sample_rate) for t in targets]
        except Exception as exc:
            raise HTTPException(400, f"Bad input: {exc}") from exc

        with gpu_lock:
            start = time.perf_counter()
            try:
                out = adapter.convert(task, src, tgts, parsed)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                log.error("convert failed for %s\n%s", name, traceback.format_exc())
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
            compute = time.perf_counter() - start

        return Response(
            write_wav(out, adapter.output_sample_rate),
            media_type="audio/wav",
            headers={"X-Compute-Seconds": f"{compute:.3f}"},
        )

    return app
