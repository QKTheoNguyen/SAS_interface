"""Run every (model, task) once through a running gateway, with manifest defaults.

    uv run --project gateway python scripts/smoke_test.py [--url http://127.0.0.1:8000]

Waits for all models to finish loading, then converts a bundled LibriSpeech clip
and writes the outputs to logs/smoke/. Exit code 1 if any run fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx
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
                if "non-streaming" not in model["modes"]:
                    continue
                targets = TARGETS if spec.get("target_voice", "none") == "required" else []
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
