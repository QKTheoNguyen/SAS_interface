from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from .adapter import ModelAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def load_manifests(models_dir: Path = DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(models_dir.glob("*/manifest.yaml")):
        with path.open() as f:
            manifest = yaml.safe_load(f)
        manifest["_dir"] = str(path.parent)
        manifests.append(manifest)
    return manifests


def load_adapter(manifest: dict[str, Any]) -> ModelAdapter:
    """Import ``models/<name>/adapter.py`` and instantiate its ``Adapter`` class."""
    adapter_path = Path(manifest["_dir"]) / "adapter.py"
    module_name = f"sas_models.{manifest['name']}"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Adapter(manifest)


def adapters_for_worker(worker: str, models_dir: Path = DEFAULT_MODELS_DIR) -> list[ModelAdapter]:
    return [load_adapter(m) for m in load_manifests(models_dir) if m["worker"] == worker]
