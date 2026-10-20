"""Gateway entry point.

    python -m sas_gateway serve           # start the web gateway
    python -m sas_gateway local-workers   # print "name host port" per worker run.sh must start
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="sas_gateway")
    parser.add_argument("command", choices=["serve", "local-workers"])
    parser.add_argument("--config", default=None, help="Default: $SAS_CONFIG or <repo>/config.yaml")
    args = parser.parse_args()

    from .config import load_config

    config = load_config(args.config)

    if args.command == "local-workers":
        for name, host, port in config.local_workers():
            print(name, host, port)
        return

    import uvicorn

    from .app import create_app

    app = create_app(config, on_ready=lambda: print(f"Gateway ready on http://{config.host}:{config.port}", flush=True))
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
