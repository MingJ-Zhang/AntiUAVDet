from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from .io import atomic_json, load_json

VALID_STATES = {"PENDING", "RUNNING", "FAILED", "COMPLETED"}


def update_manifest(path: str | Path, model: str, state: str, **extra: object) -> dict:
    if state not in VALID_STATES:
        raise ValueError(f"invalid state: {state}")
    manifest_path = Path(path)
    manifest = load_json(manifest_path, {"schema_version": 1, "models": {}})
    current = manifest["models"].get(model, {})
    history = current.get("history", [])
    event = {
        "state": state,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        **extra,
    }
    history.append(event)
    merged = {**current}
    if state in {"PENDING", "RUNNING"}:
        merged.pop("exit_code", None)
    if state == "PENDING":
        merged.pop("resume_point", None)
        merged.pop("peak_memory_mib", None)
    manifest["models"][model] = {**merged, **event, "history": history}
    atomic_json(manifest_path, manifest)
    return manifest
