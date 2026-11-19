"""Worker entry point.

    python -m sas_worker serve    --worker quickconvert --port 8101
    python -m sas_worker prefetch --worker quickconvert

Two worker names are special and serve a service instead of models from models/*/manifest.yaml:
``metrics`` (the metrics panel, sas_worker/metrics.py) and ``pii`` (PII detection and
obfuscation, sas_worker/pii_server.py).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .registry import DEFAULT_MODELS_DIR, adapters_for_worker

METRICS_WORKER = "metrics"
PII_WORKER = "pii"


def main() -> None:
    parser = argparse.ArgumentParser(prog="sas_worker")
    parser.add_argument("command", choices=["serve", "prefetch"])
    parser.add_argument("--worker", required=True, help="Worker name, as in manifest.yaml `worker:` (or `metrics` / `pii`)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{args.worker}] %(levelname)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.worker == METRICS_WORKER:
        from . import metrics

        if args.command == "prefetch":
            metrics.prefetch()
            return
        import uvicorn

        uvicorn.run(metrics.create_metrics_app(device), host=args.host, port=args.port)
        return

    if args.worker == PII_WORKER:
        from . import pii_server

        if args.command == "prefetch":
            pii_server.prefetch()
            return
        import uvicorn

        uvicorn.run(pii_server.create_pii_app(device), host=args.host, port=args.port)
        return

    adapters = adapters_for_worker(args.worker, args.models_dir)
    if not adapters:
        raise SystemExit(f"No model in {args.models_dir} declares `worker: {args.worker}`")

    if args.command == "prefetch":
        for adapter in adapters:
            logging.info("Prefetching %s", adapter.name)
            adapter.prefetch()
        return

    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(args.worker, adapters, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
