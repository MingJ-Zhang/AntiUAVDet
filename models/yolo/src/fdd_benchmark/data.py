from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

from PIL import Image

from .config import BenchmarkConfig
from .io import atomic_json, sha256_file

EXPECTED = {
    "train": {"images": 4599, "annotations": 4544, "empty": 123},
    "val": {"images": 1438, "annotations": 1424, "empty": 28},
    "test": {"images": 1150, "annotations": 1117, "empty": 45},
}
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _inventory_sha256(paths: list[Path], base: Path) -> str:
    """对相对路径和文件内容做稳定聚合哈希，避免只锁定标注而漏掉图像漂移。"""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(target, target_is_directory=True)


def _configured_expected(cfg: BenchmarkConfig) -> dict[str, dict[str, int]]:
    configured = cfg.raw.get("dataset_expected")
    if configured is None:
        return EXPECTED
    return {
        split: {key: int(value) for key, value in configured[split].items()}
        for split in SPLITS
    }


def _materialize_labels(cfg: BenchmarkConfig, split: str) -> list[dict[str, object]]:
    """复制并修复派生标签视图，永不改动源数据。"""
    source = cfg.label_dir(split)
    destination = cfg.dataset_view / "labels" / split
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o755)
    repair_policy = cfg.raw.get("dataset_repairs", {})
    drop_non_positive = bool(repair_policy.get("drop_non_positive_boxes", False))
    clip_out_of_bounds = bool(repair_policy.get("clip_out_of_bounds_boxes", False))
    repairs: list[dict[str, object]] = []
    source_names = {path.name for path in source.glob("*.txt")}
    for stale in destination.glob("*.txt"):
        if stale.name not in source_names:
            stale.chmod(0o644)
            stale.unlink()
    for label_path in sorted(source.glob("*.txt")):
        kept: list[str] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            should_drop = False
            values: list[float] | None = None
            if len(fields) == 5:
                with contextlib.suppress(ValueError):
                    values = [float(value) for value in fields]
                    should_drop = drop_non_positive and (values[3] <= 0 or values[4] <= 0)
            if should_drop:
                repairs.append(
                    {
                        "split": split,
                        "file": label_path.name,
                        "line": line_number,
                        "action": "drop_non_positive_box",
                        "source": line,
                    }
                )
            elif values is not None and clip_out_of_bounds:
                class_id, center_x, center_y, width, height = values
                x1, y1 = center_x - width / 2.0, center_y - height / 2.0
                x2, y2 = center_x + width / 2.0, center_y + height / 2.0
                clipped = (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
                if clipped != (x1, y1, x2, y2):
                    x1, y1, x2, y2 = clipped
                    repaired = (
                        f"{int(class_id)} {(x1 + x2) / 2.0:.10g} {(y1 + y2) / 2.0:.10g} "
                        f"{x2 - x1:.10g} {y2 - y1:.10g}"
                    )
                    repairs.append(
                        {
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                            "action": "clip_box_to_image_bounds",
                            "source": line,
                            "derived": repaired,
                        }
                    )
                    kept.append(repaired)
                else:
                    kept.append(line)
            else:
                kept.append(line)
        output = destination / label_path.name
        if output.exists():
            output.chmod(0o644)
        output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        output.chmod(0o444)
    destination.chmod(0o555)
    return repairs


def _build_coco_annotations(cfg: BenchmarkConfig, split: str) -> None:
    if not cfg.raw.get("generate_coco_from_yolo", False):
        return
    image_dir = cfg.image_dir(split)
    label_dir = cfg.training_label_dir(split)
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco_images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, center_x, center_y, box_width, box_height = map(float, line.split())
            width_px = box_width * width
            height_px = box_height * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [
                        center_x * width - width_px / 2.0,
                        center_y * height - height_px / 2.0,
                        width_px,
                        height_px,
                    ],
                    "area": width_px * height_px,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "info": {"description": "Deterministic COCO view derived from source YOLO labels"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": cfg.raw.get("class_names", ["drone"])[0]}],
    }
    atomic_json(cfg.annotation(split), payload)


def prepare_dataset_view(cfg: BenchmarkConfig) -> dict:
    view = cfg.dataset_view
    view.mkdir(parents=True, exist_ok=True)
    view.chmod(0o755)
    for kind in ("images", "labels", "lists"):
        (view / kind).mkdir(parents=True, exist_ok=True)
        (view / kind).chmod(0o755)
    annotation_alias = cfg.raw.get("yolov5_annotation_alias_pattern")
    if annotation_alias:
        (view / "annotations").mkdir(parents=True, exist_ok=True)
        (view / "annotations").chmod(0o755)
    repairs: list[dict[str, object]] = []
    materialize_labels = cfg.raw.get("dataset_layout", {}).get("materialize_labels", False)
    for split in SPLITS:
        _replace_symlink(view / "images" / split, cfg.image_dir(split))
        if materialize_labels:
            repairs.extend(_materialize_labels(cfg, split))
        else:
            _replace_symlink(view / "labels" / split, cfg.label_dir(split))
        images = sorted(
            path
            for path in (view / "images" / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_list = view / "lists" / f"{split}.txt"
        if image_list.exists():
            image_list.chmod(0o644)
        image_list.write_text(
            "\n".join(str(view / "images" / split / path.name) for path in images) + "\n",
            encoding="utf-8",
        )
        image_list.chmod(0o444)
        _build_coco_annotations(cfg, split)
        if annotation_alias:
            alias = view / "annotations" / annotation_alias.format(split=split)
            if alias.exists():
                alias.chmod(0o644)
            shutil.copy2(cfg.annotation(split), alias)
            alias.chmod(0o444)
    managed_kinds = ["images", "labels", "lists"]
    if annotation_alias:
        managed_kinds.append("annotations")
    for kind in managed_kinds:
        (view / kind).chmod(0o555)
    view.chmod(0o555)
    report = validate_dataset(cfg)
    report["repairs"] = repairs
    atomic_json(cfg.runtime_root / "dataset_report.json", report)
    return report


def validate_dataset(cfg: BenchmarkConfig) -> dict:
    report: dict[str, object] = {"dataset_root": str(cfg.dataset_root), "splits": {}}
    for split, expected in _configured_expected(cfg).items():
        image_dir = cfg.image_dir(split)
        label_dir = cfg.training_label_dir(split)
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        if set(images) != set(labels):
            raise RuntimeError(f"{split}: image/label stems differ")
        invalid: list[str] = []
        empty = 0
        label_boxes = 0
        for _stem, label_path in labels.items():
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    invalid.append(f"{label_path}:{line_number}: field-count")
                    continue
                values = [float(value) for value in fields]
                if not all(math.isfinite(value) for value in values):
                    invalid.append(f"{label_path}:{line_number}: non-finite")
                if int(values[0]) != 0 or values[0] != 0:
                    invalid.append(f"{label_path}:{line_number}: class")
                if not all(0.0 <= value <= 1.0 for value in values[1:]):
                    invalid.append(f"{label_path}:{line_number}: coordinate")
                if values[3] <= 0 or values[4] <= 0:
                    invalid.append(f"{label_path}:{line_number}: non-positive box")
                label_boxes += 1
        if invalid:
            raise RuntimeError("\n".join(invalid[:20]))
        coco_path = cfg.annotation(split)
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        if {int(item["id"]) for item in coco.get("categories", [])} != {0}:
            raise RuntimeError(f"{split}: COCO categories must contain only id=0")
        actual = {"images": len(images), "annotations": len(coco["annotations"]), "empty": empty}
        if actual != expected or label_boxes != expected["annotations"]:
            raise RuntimeError(
                f"{split}: expected={expected}, actual={actual}, label_boxes={label_boxes}"
            )
        coco_names = {Path(item["file_name"]).stem for item in coco["images"]}
        if coco_names != set(images):
            raise RuntimeError(f"{split}: COCO/image stems differ")
        image_list = cfg.dataset_view / "lists" / f"{split}.txt"
        listed = image_list.read_text(encoding="utf-8").splitlines()
        expected_list = [
            str(cfg.dataset_view / "images" / split / path.name) for path in sorted(images.values())
        ]
        if listed != expected_list:
            raise RuntimeError(f"{split}: dataset image list differs from source inventory")
        if image_list.stat().st_mode & 0o222:
            raise RuntimeError(f"{split}: dataset image list must be read-only")
        image_by_id = {int(item["id"]): item for item in coco["images"]}
        if len(image_by_id) != len(coco["images"]):
            raise RuntimeError(f"{split}: duplicate COCO image ids")
        for item in coco["images"]:
            image_path = images[Path(item["file_name"]).stem]
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (int(item["width"]), int(item["height"])):
                    raise RuntimeError(f"{split}/{image_path.name}: COCO image dimensions differ")
        coco_boxes: dict[str, list[tuple[float, ...]]] = {stem: [] for stem in images}
        for item in coco["annotations"]:
            if int(item["category_id"]) != 0:
                raise RuntimeError(f"{split}: COCO annotation category must be 0")
            if int(item["image_id"]) not in image_by_id:
                raise RuntimeError(f"{split}: COCO annotation references unknown image")
            image = image_by_id[int(item["image_id"])]
            x, y, width, height = (float(value) for value in item["bbox"])
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise RuntimeError(f"{split}: non-finite COCO bbox")
            tolerance = 1e-5
            if (
                x < -tolerance
                or y < -tolerance
                or width <= 0
                or height <= 0
                or x + width > float(image["width"]) + tolerance
                or y + height > float(image["height"]) + tolerance
            ):
                raise RuntimeError(f"{split}: out-of-bounds COCO bbox")
            normalized = (
                0.0,
                (x + width / 2.0) / float(image["width"]),
                (y + height / 2.0) / float(image["height"]),
                width / float(image["width"]),
                height / float(image["height"]),
            )
            coco_boxes[Path(image["file_name"]).stem].append(normalized)
        for stem, label_path in labels.items():
            yolo_boxes = [
                tuple(float(value) for value in line.split())
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            expected_boxes = coco_boxes[stem]
            if len(yolo_boxes) != len(expected_boxes):
                raise RuntimeError(f"{split}/{stem}: COCO/YOLO box counts differ")
            yolo_boxes.sort()
            expected_boxes.sort()
            for yolo_box, coco_box in zip(yolo_boxes, expected_boxes, strict=True):
                difference = zip(yolo_box, coco_box, strict=True)
                if any(abs(first - second) > 2e-5 for first, second in difference):
                    raise RuntimeError(f"{split}/{stem}: COCO/YOLO coordinates differ")
        source_label_dir = cfg.label_dir(split)
        report["splits"][split] = {
            **actual,
            "coco_sha256": sha256_file(coco_path),
            "images_sha256": _inventory_sha256(list(images.values()), image_dir),
            "labels_sha256": _inventory_sha256(list(labels.values()), label_dir),
            "source_labels_sha256": _inventory_sha256(
                list(source_label_dir.glob("*.txt")), source_label_dir
            ),
            "image_list_sha256": sha256_file(image_list),
            "image_dir": str(image_dir),
        }
    if cfg.raw.get("audit_exact_duplicates", False):
        groups: dict[str, list[dict[str, str]]] = {}
        for split in SPLITS:
            for image_path in sorted(cfg.image_dir(split).iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                groups.setdefault(sha256_file(image_path), []).append(
                    {"split": split, "file": image_path.name}
                )
        repeated = [items for items in groups.values() if len(items) > 1]
        cross_split = [items for items in repeated if len({item["split"] for item in items}) > 1]
        within_split = [items for items in repeated if len({item["split"] for item in items}) == 1]
        report["exact_duplicate_audit"] = {
            "unique_image_hashes": len(groups),
            "cross_split_groups": cross_split,
            "within_split_groups": within_split,
        }
    runtime = cfg.runtime_root
    runtime.mkdir(parents=True, exist_ok=True)
    atomic_json(runtime / "dataset_report.json", report)
    return report


def assert_read_only_source(cfg: BenchmarkConfig) -> None:
    """Reject the common accidental case where the view resolves inside the project."""
    if os.path.commonpath([cfg.dataset_root, cfg.project_root]) == str(cfg.project_root):
        raise RuntimeError("dataset source must remain outside the project tree")
