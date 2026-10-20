"""Worker entry point.

    python -m sas_worker serve    --worker quickconvert --port 8101
    python -m sas_worker prefetch --worker quickconvert
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .registry import DEFAULT_MODELS_DIR, adapters_for_worker


def main() -> None:
    parser = argparse.ArgumentParser(prog="sas_worker")
    parser.add_argument("command", choices=["serve", "prefetch"])
    parser.add_argument("--worker", required=True, help="Worker name, as in manifest.yaml `worker:`")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{args.worker}] %(levelname)s %(message)s")
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    uvicorn.run(create_app(args.worker, adapters, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
