from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    max_input_seconds: float = 60.0
    max_stream_seconds: float = 300.0
    workers: dict[str, str] = field(default_factory=dict)  # name -> base URL
    models_dir: Path = Path(os.environ.get("SAS_MODELS_DIR", REPO_ROOT / "models"))

    def local_workers(self) -> list[tuple[str, str, int]]:
        """(name, host, port) of workers that run.sh must start on this machine."""
        out = []
        for name, url in self.workers.items():
            parsed = urlparse(url)
            if parsed.hostname in LOCAL_HOSTS:
                out.append((name, parsed.hostname, parsed.port or 80))
        return out


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("SAS_CONFIG", REPO_ROOT / "config.yaml"))
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    gateway = raw.get("gateway") or {}
    workers = {name: str(spec["url"]).rstrip("/") for name, spec in (raw.get("workers") or {}).items()}
    return Config(
        host=os.environ.get("SAS_HOST", gateway.get("host", "127.0.0.1")),
        port=int(os.environ.get("SAS_PORT", gateway.get("port", 8000))),
        max_input_seconds=float(gateway.get("max_input_seconds", 60)),
        max_stream_seconds=float(gateway.get("max_stream_seconds", 300)),
        workers=workers,
    )


def load_manifests(config: Config) -> dict[str, dict[str, Any]]:
    """Manifests of models whose worker is enabled in the config, keyed by model name."""
    manifests = {}
    for path in sorted(config.models_dir.glob("*/manifest.yaml")):
        with path.open() as f:
            manifest = yaml.safe_load(f)
        if manifest["worker"] in config.workers:
            manifests[manifest["name"]] = manifest
    return manifests
