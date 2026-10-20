"""Shared code for SAS_interface model workers.

A worker is one process, in one Python environment, serving one or more model
adapters over HTTP. Each model lives in ``models/<name>/`` with a
``manifest.yaml`` (read by the gateway and the worker) and an ``adapter.py``
(imported only by the worker named in the manifest).
"""

from .adapter import ModelAdapter, StreamSession

__all__ = ["ModelAdapter", "StreamSession"]
