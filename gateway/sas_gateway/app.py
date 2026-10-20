from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import Config, load_config, load_manifests

STATIC_DIR = Path(__file__).parent / "static"
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
        return {"max_input_seconds": config.max_input_seconds}

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
            out.append({**manifest, "status": status, "error": error, "device": (health or {}).get("device")})
        return out

    @app.get("/api/models/{name}/choices/{param}")
    async def choices(name: str, param: str) -> Response:
        manifest = manifest_or_404(name)
        try:
            r = await client.get(f"{config.workers[manifest['worker']]}/models/{name}/choices/{param}", timeout=10.0)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Worker unreachable: {exc}") from exc
        return await forward(r)

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

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
