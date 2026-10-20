"""Run every (model, task) once through a running gateway, with manifest defaults.

    uv run --project gateway python scripts/smoke_test.py [--url http://127.0.0.1:8000]

Waits for all models to finish loading, then converts a bundled LibriSpeech clip
and writes the outputs to logs/smoke/. Exit code 1 if any run fails.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import struct
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "models/streamvoiceanon/references"
SOURCE = REFS / "M_1571.flac"
TARGETS = [REFS / "F_192.flac", REFS / "F_3513.flac", REFS / "F_7754.flac"]  # same-ish total ~48 s for kNN-VC


def default_params(spec: dict) -> dict:
    params = {p["name"]: p.get("default") for p in spec.get("params", [])}
    preset = next((p for p in spec.get("presets", []) if p.get("default")), None)
    params.update((preset or {}).get("values", {}))
    return params


def wait_ready(client: httpx.Client, timeout: float) -> list[dict]:
    deadline = time.time() + timeout
    while True:
        try:
            models = client.get("/api/models").json()
            pending = [m["name"] for m in models if m["status"] in ("pending", "loading", "offline")]
            if not pending:
                return models
            print(f"waiting for: {', '.join(pending)}", flush=True)
        except httpx.HTTPError as exc:
            print(f"gateway not up yet ({exc.__class__.__name__})", flush=True)
        if time.time() > deadline:
            sys.exit("Timed out waiting for models to load")
        time.sleep(5)


async def stream_once(base_url: str, model: str, task: str, params: dict, targets: list[Path]) -> dict:
    """Stream SOURCE through the gateway at real-time pace, like the browser does."""
    from websockets.asyncio.client import connect

    audio, sr = sf.read(SOURCE, dtype="float32")
    url = base_url.replace("http", "ws", 1) + f"/api/models/{model}/stream"
    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "task": task, "params": params, "n_targets": len(targets)}))
        for t in targets:
            await ws.send(t.read_bytes())
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise RuntimeError(ready.get("detail"))
        rate, n = ready["sample_rate"], ready["chunk_samples"]
        audio = np.interp(np.arange(0, len(audio), sr / rate), np.arange(len(audio)), audio).astype("<f4")
        chunks = [audio[i : i + n] for i in range(0, len(audio) - n + 1, n)]
        flush = int(np.ceil(ready["algorithmic_latency_ms"] / (n / rate * 1000))) + 2
        chunks += [np.zeros(n, "<f4")] * flush

        sent: dict[int, float] = {}
        out, compute, round_trip = [], [], []

        async def sender():
            start = time.perf_counter()
            for seq, chunk in enumerate(chunks, 1):
                await asyncio.sleep(max(0.0, start + (seq - 1) * n / rate - time.perf_counter()))
                sent[seq] = time.perf_counter()
                await ws.send(struct.pack("<I", seq) + chunk.tobytes())
            await ws.send(json.dumps({"type": "stop"}))

        async def receiver():
            async for message in ws:
                if isinstance(message, str):
                    msg = json.loads(message)
                    if msg["type"] == "error":
                        raise RuntimeError(msg["detail"])
                    return
                seq, ms = struct.unpack_from("<If", message)
                round_trip.append((time.perf_counter() - sent[seq]) * 1000)
                compute.append(ms)
                out.append(np.frombuffer(message, "<f4", offset=8))

        await asyncio.gather(sender(), receiver())
    output = np.concatenate(out)
    chunk_ms = n / rate * 1000
    return {
        "ready": ready,
        "chunks": len(compute),
        "compute_ms": float(np.mean(compute[3:] or compute)),  # skip the delay warm-up chunks
        "rtf": float(np.mean(compute[3:] or compute)) / chunk_ms,
        "round_trip_ms": float(np.median(round_trip)),
        "output": output,
        "rate": ready["output_sample_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--load-timeout", type=float, default=900)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "logs/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(base_url=args.url, timeout=600) as client:
        for model in wait_ready(client, args.load_timeout):
            for task, spec in model["tasks"].items():
                label = f"{model['name']:16} {task:17}"
                if model["status"] != "ready":
                    print(f"FAIL {label} model {model['status']}: {model['error']}")
                    failures += 1
                    continue
                targets = TARGETS if spec.get("target_voice", "none") == "required" else []
                if "streaming" in model["modes"]:
                    try:
                        r = asyncio.run(stream_once(args.url, model["name"], task, default_params(spec), targets))
                    except Exception as exc:
                        print(f"FAIL {label} streaming: {type(exc).__name__}: {exc}")
                        failures += 1
                    else:
                        path = out_dir / f"{model['name']}_{task}_streaming.wav"
                        sf.write(path, r["output"], r["rate"])
                        rms = float(np.sqrt(np.mean(r["output"] ** 2)))
                        print(
                            f"{'OK  ' if rms > 1e-3 else 'FAIL'} {label} streaming: {r['chunks']} chunks, compute "
                            f"{r['compute_ms']:.1f} ms/chunk, RTF {r['rtf']:.2f}, round trip {r['round_trip_ms']:.0f} ms, "
                            f"model delay {r['ready']['algorithmic_latency_ms']:.0f} ms, rms {rms:.3f} -> {path.relative_to(REPO_ROOT)}"
                        )
                        failures += rms <= 1e-3
                if "non-streaming" not in model["modes"]:
                    continue
                files = [("source", (SOURCE.name, SOURCE.read_bytes(), "audio/flac"))]
                files += [("targets", (t.name, t.read_bytes(), "audio/flac")) for t in targets]
                data = {"task": task, "params": json.dumps(default_params(spec))}
                start = time.time()
                r = client.post(f"/api/models/{model['name']}/convert", data=data, files=files)
                if r.is_error:
                    print(f"FAIL {label} HTTP {r.status_code}: {r.text[:300]}")
                    failures += 1
                    continue
                info = sf.info(io.BytesIO(r.content))
                path = out_dir / f"{model['name']}_{task}.wav"
                path.write_bytes(r.content)
                print(
                    f"OK   {label} model {float(r.headers.get('x-compute-seconds', 'nan')):6.2f} s, "
                    f"total {time.time() - start:6.2f} s, output {info.duration:5.1f} s @ {info.samplerate} Hz -> {path.relative_to(REPO_ROOT)}"
                )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
