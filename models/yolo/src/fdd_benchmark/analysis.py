from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import atomic_json


def box_iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _ground_truth(annotation: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in annotation["annotations"]:
        grouped[int(item["image_id"])].append(item)
    return grouped


def precision_recall_curve(
    annotation_path: str | Path,
    prediction_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, list[float] | float]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    predictions = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    ground_truth = _ground_truth(annotation)
    used: dict[int, set[int]] = defaultdict(set)
    ordered = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    true_positive: list[float] = []
    false_positive: list[float] = []
    for prediction in ordered:
        image_id = int(prediction["image_id"])
        candidates = ground_truth.get(image_id, [])
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(candidates):
            if index in used[image_id]:
                continue
            iou = box_iou_xywh(prediction["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou, best_index = iou, index
        matched = best_index >= 0 and best_iou >= iou_threshold
        if matched:
            used[image_id].add(best_index)
        true_positive.append(float(matched))
        false_positive.append(float(not matched))
    if not ordered:
        return {
            "thresholds": [1.0],
            "precision": [0.0],
            "recall": [0.0],
            "f1": [0.0],
            "best_threshold": 1.0,
            "best_f1": 0.0,
        }
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(len(annotation["annotations"]), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1))
    return {
        "thresholds": [float(item["score"]) for item in ordered],
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "best_threshold": float(ordered[best_index]["score"]),
        "best_f1": float(f1[best_index]),
    }


def confusion_counts(
    annotation_path: str | Path,
    prediction_path: str | Path,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, int | float]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    predictions = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    ground_truth = _ground_truth(annotation)
    by_image: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction["score"]) >= score_threshold:
            by_image[int(prediction["image_id"])].append(prediction)
    true_positive = false_positive = false_negative = 0
    image_ids = {int(item["id"]) for item in annotation["images"]}
    for image_id in image_ids:
        used: set[int] = set()
        ordered = sorted(
            by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True
        )
        for prediction in ordered:
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(ground_truth.get(image_id, [])):
                if index in used:
                    continue
                iou = box_iou_xywh(prediction["bbox"], target["bbox"])
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(ground_truth.get(image_id, [])) - len(used)
    return {
        "score_threshold_from_val": score_threshold,
        "iou_threshold": iou_threshold,
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
    }


def analyze_predictions(
    val_annotation: Path,
    val_predictions: Path,
    test_annotation: Path,
    test_predictions: Path,
    output_dir: Path,
) -> dict:
    curve = precision_recall_curve(val_annotation, val_predictions)
    threshold = float(curve["best_threshold"])
    result = {
        "threshold_selection": {
            "split": "val",
            "criterion": "maximum F1 at IoU=0.50",
            "score_threshold": threshold,
            "best_f1": curve["best_f1"],
        },
        "val": confusion_counts(val_annotation, val_predictions, threshold),
        "test": confusion_counts(test_annotation, test_predictions, threshold),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "pr_curve.json", curve)
    atomic_json(output_dir / "operating_point.json", result)
    return result
