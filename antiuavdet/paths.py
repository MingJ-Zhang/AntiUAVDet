"""Repository layout and environment paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_ENV = "ANTIUAVDET_DATA"
WEIGHTS_ENV = "ANTIUAVDET_WEIGHTS"

DATASET_KEYS = {
    "fdd": "FDD",
    "antiuav": "antiuav",
    "dut_dve": "DUT-Dve_YOLO_Benchmark_View",
}


def data_root() -> Path:
    raw = os.environ.get(DATA_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "data" / "datasets").resolve()


def expand_vars(text: str) -> str:
    data = str(data_root()).replace("\\", "/")
    weights = str(weights_root()).replace("\\", "/")
    return (
        text.replace("${ANTIUAVDET_DATA}", data)
        .replace("${ANTIUAVDET_WEIGHTS}", weights)
    )


def materialize_config(src: Path) -> Path:
    """Write a temp copy with ${ANTIUAVDET_DATA} expanded (YAML trainers)."""
    text = expand_vars(src.read_text(encoding="utf-8"))
    dest = ROOT / ".cache" / "configs" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def weights_root() -> Path:
    raw = os.environ.get(WEIGHTS_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "weights").resolve()


def dataset_dir(dataset: str) -> Path:
    key = DATASET_KEYS.get(dataset, dataset)
    return data_root() / key


def img_dir(dataset: str) -> Path:
    """Default test-image directories used in the published benchmark."""
    mapping = {
        "fdd": dataset_dir("fdd") / "test" / "foggy",
        "antiuav": dataset_dir("antiuav") / "test" / "test" / "img",
        "dut_dve": dataset_dir("dut_dve") / "images" / "test",
    }
    return mapping[dataset]


def gt_path(dataset: str) -> Path:
    return ROOT / "data" / "gt" / f"{dataset}_test.json"


def model_dir(rel: str) -> Path:
    return (ROOT / rel).resolve()
