"""Load configs/model_zoo.yaml — the single source of truth for detectors."""
from __future__ import annotations

from pathlib import Path

import yaml

from .paths import ROOT, weights_root


def zoo_path() -> Path:
    return ROOT / "configs" / "model_zoo.yaml"


def load_zoo(path: Path | None = None) -> dict:
    p = path or zoo_path()
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def datasets(zoo: dict | None = None) -> list[str]:
    z = zoo or load_zoo()
    return list(z.get("datasets", ["fdd", "antiuav", "dut_dve"]))


def models(zoo: dict | None = None) -> list[dict]:
    z = zoo or load_zoo()
    return list(z.get("models", []))


def get_model(model_id: str, zoo: dict | None = None) -> dict:
    for m in models(zoo):
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model '{model_id}'. Run: antiuavdet list")


def active_datasets(model: dict, zoo: dict | None = None) -> list[str]:
    skip = set(model.get("skip") or [])
    return [d for d in datasets(zoo) if d not in skip]


def repo_path(model: dict) -> Path:
    return ROOT / model["repo"]


def config_path(model: dict, dataset: str) -> Path:
    rel = (model.get("configs") or {}).get(dataset)
    if not rel:
        raise KeyError(f"{model['id']} has no config for dataset '{dataset}'")
    return repo_path(model) / rel


def weight_path(model: dict, dataset: str) -> Path:
    """User-supplied checkpoint. Default: weights/<id>/<dataset>.<ext>"""
    rel = (model.get("ckpts") or {}).get(dataset)
    if rel:
        candidate = repo_path(model) / rel
        if candidate.is_file():
            return candidate
        # also accept the documented weights/ layout
    ext = ".pt" if model["framework"] in {"ultralytics"} else ".pth"
    return weights_root() / model["id"] / f"{dataset}{ext}"
