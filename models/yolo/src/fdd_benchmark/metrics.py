from __future__ import annotations

import contextlib
import csv
import json
import math
import re
import shutil
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .io import atomic_json

METRIC_NAMES = [
    "AP50_95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def normalize_predictions(
    raw_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    image_id_semantics: str = "filename_stem",
) -> list[dict]:
    if image_id_semantics not in {"filename_stem", "coco_id"}:
        raise ValueError(f"unsupported image id semantics: {image_id_semantics}")
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    stem_to_id: dict[object, int] = {}
    valid_ids = {int(item["id"]) for item in annotation["images"]}
    for image in annotation["images"]:
        stem = Path(image["file_name"]).stem
        stem_to_id[stem] = int(image["id"])
        if stem.isnumeric():
            stem_to_id[int(stem)] = int(image["id"])
    normalized: list[dict] = []
    for prediction in raw:
        raw_id = prediction["image_id"]
        mapped = None
        if image_id_semantics == "coco_id":
            with contextlib.suppress(TypeError, ValueError):
                candidate = int(raw_id)
                mapped = candidate if candidate in valid_ids else None
        else:
            mapped = stem_to_id.get(raw_id)
            if mapped is None and isinstance(raw_id, str):
                mapped = stem_to_id.get(Path(raw_id).stem)
        if mapped is None:
            raise KeyError(f"cannot map prediction image_id={raw_id!r} as {image_id_semantics}")
        bbox = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError(f"invalid prediction bbox: {bbox}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid prediction score: {score}")
        normalized.append(
            {
                "image_id": mapped,
                "category_id": 0,
                "bbox": bbox,
                "score": score,
            }
        )
    atomic_json(output_path, normalized)
    return normalized


def evaluate_coco(annotation_path: str | Path, prediction_path: str | Path) -> dict[str, float]:
    coco_gt = COCO(str(annotation_path))
    predictions = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": coco_gt.dataset["images"],
            "categories": coco_gt.dataset["categories"],
            "annotations": [],
        }
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        name: float(value) * 100.0
        for name, value in zip(METRIC_NAMES, evaluator.stats, strict=True)
    }


def evaluate_prediction_file(
    raw_path: Path,
    annotation: Path,
    normalized_path: Path,
    metrics_path: Path,
    image_id_semantics: str = "filename_stem",
) -> dict:
    normalize_predictions(
        raw_path,
        annotation,
        normalized_path,
        image_id_semantics=image_id_semantics,
    )
    metrics = evaluate_coco(annotation, normalized_path)
    atomic_json(metrics_path, metrics)
    return metrics


def select_best_epoch(
    run_dir: str | Path,
    annotation_path: str | Path,
    selected_output: str | Path,
    expected_epochs: int | None = None,
) -> dict:
    run = Path(run_dir)
    raw_files = sorted((run / "val_predictions").glob("epoch_*.raw.json"))
    if not raw_files:
        raise FileNotFoundError(f"no per-epoch predictions in {run / 'val_predictions'}")
    if expected_epochs is not None and len(raw_files) != expected_epochs:
        raise RuntimeError(
            f"formal run incomplete: expected {expected_epochs} epoch predictions, "
            f"found {len(raw_files)}"
        )
    rows: list[dict] = []
    normalized_dir = run / "val_predictions" / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in raw_files:
        match = re.search(r"epoch_(\d+)", raw_path.stem)
        if not match:
            continue
        epoch = int(match.group(1))
        normalized = normalized_dir / f"epoch_{epoch:03d}.coco.json"
        normalize_predictions(raw_path, annotation_path, normalized)
        rows.append({"epoch": epoch, **evaluate_coco(annotation_path, normalized)})
    rows.sort(key=lambda row: row["epoch"])
    best = max(rows, key=lambda row: (row["AP50_95"], -row["epoch"]))
    csv_path = run / "val_epoch_coco_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = run / "train" / "weights" / f"epoch{best['epoch']}.pt"
    if not checkpoint.exists() and best["epoch"] == max(row["epoch"] for row in rows):
        checkpoint = run / "train" / "weights" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint for epoch {best['epoch']} not found")
    selected = Path(selected_output)
    selected.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, selected)
    summary = {
        "best_epoch": int(best["epoch"]),
        "best_val": {key: value for key, value in best.items() if key != "epoch"},
        "last10_mean_AP50_95": sum(row["AP50_95"] for row in rows[-10:]) / min(10, len(rows)),
        "selected_checkpoint": str(selected),
        "source_checkpoint": str(checkpoint),
    }
    atomic_json(run / "selection.json", summary)
    return summary
