from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("sas_worker.download")


def fetch(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless it already exists. The file appears only once complete."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
    Path(tmp.name).replace(path)
    return path


def fetch_zip(url: str, extract_to: Path, marker: Path) -> None:
    """Download and unpack a zip archive into ``extract_to``, unless ``marker`` (a path the archive creates) exists."""
    if Path(marker).exists():
        return
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=extract_to) as tmp:
        archive = fetch(url, Path(tmp) / "archive.zip")
        log.info("Unpacking %s into %s", archive.name, extract_to)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_to)
    if not Path(marker).exists():
        raise FileNotFoundError(f"{url} did not contain the expected {marker}")
