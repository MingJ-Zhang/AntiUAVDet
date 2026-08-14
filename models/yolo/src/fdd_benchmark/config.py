from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import atomic_json, load_json, sha256_file


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(self.raw["project_root_remote"])

    @property
    def experiment_root(self) -> Path:
        """实验产物根目录；默认保持原 FDD 项目的历史布局。"""
        return Path(self.raw.get("experiment_root_remote", self.raw["project_root_remote"]))

    @property
    def runtime_root(self) -> Path:
        return self.experiment_root / "runtime"

    @property
    def runs_root(self) -> Path:
        return self.experiment_root / "runs"

    @property
    def artifacts_root(self) -> Path:
        return self.experiment_root / "artifacts"

    @property
    def dataset_root(self) -> Path:
        return Path(self.raw["dataset_root"])

    @property
    def dataset_view(self) -> Path:
        return Path(self.raw["dataset_view"])

    @property
    def annotation_root(self) -> Path:
        return Path(self.raw["annotation_root"])

    @property
    def protocol(self) -> dict[str, Any]:
        return self.raw["protocol"]

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self.raw["models"]

    @property
    def input_shape(self) -> tuple[int, int]:
        """协议固定输入的 (height, width)。"""
        height, width = self.protocol["image_size"]
        return int(height), int(width)

    @property
    def input_long_side(self) -> int:
        return max(self.input_shape)

    @property
    def dataset_yaml(self) -> Path:
        return self.project_root / self.raw.get("dataset_yaml", "configs/dataset/fdd.yaml")

    @property
    def yolov5_hyp(self) -> Path:
        return self.project_root / self.raw.get("yolov5_hyp", "configs/yolov5_hyp.yaml")

    def image_dir(self, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        pattern = self.raw.get("dataset_layout", {}).get("images", "{split}/foggy")
        return self.dataset_root / pattern.format(split=split)

    def label_dir(self, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        pattern = self.raw.get("dataset_layout", {}).get("labels", "{split}/labels")
        return self.dataset_root / pattern.format(split=split)

    def training_label_dir(self, split: str) -> Path:
        layout = self.raw.get("dataset_layout", {})
        if layout.get("materialize_labels", False):
            return self.dataset_view / "labels" / split
        return self.label_dir(split)

    def annotation(self, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        pattern = self.raw.get("annotation_pattern", "instances_foggy_{split}.json")
        return self.annotation_root / pattern.format(split=split)


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "project_root_remote",
        "dataset_root",
        "dataset_view",
        "annotation_root",
        "physical_gpu",
        "protocol",
        "models",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    protocol = raw["protocol"]
    expected = {
        "image_size": [768, 1344],
        "batch_size": 16,
        "epochs": 300,
        "seed": 0,
        "amp": True,
        "deterministic": True,
        "optimizer": "AdamW",
        "learning_rate": 0.0004,
        "final_lr_fraction": 1.0,
        "weight_decay": 0.0001,
        "beta1": 0.9,
        "warmup_iterations": 1000,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": 300,
        "save_period": 1,
        "conf_threshold_eval": 0.001,
        "nms_iou_threshold": 0.7,
        "max_detections": 100,
    }
    drift = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if drift:
        raise ValueError(f"formal protocol drift detected: {drift}")
    expected_augmentation = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.5,
        "copy_paste": 0.0,
        "active_start_epoch": 4,
        "active_stop_epoch": 78,
        "no_aug_start_epoch": 288,
    }
    if protocol.get("augmentation") != expected_augmentation:
        raise ValueError("formal augmentation protocol drift detected")
    height, width = protocol["image_size"]
    if height % 32 or width % 32:
        raise ValueError("formal input dimensions must both be divisible by 32")
    expected_models = {
        "yolov5n": ("yolov5", "915bbf294bb74c859f0b41f1c23bc395014ea679"),
        "yolov8n": ("ultralytics", "6340089266bfcf94d7aad2ed689a6a3db0df8883"),
        "yolo11n": ("ultralytics", "6340089266bfcf94d7aad2ed689a6a3db0df8883"),
        "yolov13n": ("yolov13", "70f23ede45ee00a30cf6139c3d1ea7abe3df4eec"),
    }
    if set(raw["models"]) != set(expected_models):
        raise ValueError("formal model registry drift detected")
    for name, (backend, revision) in expected_models.items():
        model = raw["models"][name]
        if (model.get("backend"), model.get("revision")) != (backend, revision):
            raise ValueError(f"formal source drift detected for {name}")
    if int(raw["physical_gpu"]) < 0:
        raise ValueError("physical_gpu must be a non-negative index")
    config_root = Path(raw["project_root_remote"])
    if not config_root.exists():
        config_root = config_path.parents[1]
    hyp_path = config_root / raw.get("yolov5_hyp", "configs/yolov5_hyp.yaml")
    hyp = yaml.safe_load(hyp_path.read_text(encoding="utf-8"))
    train_images = int(raw.get("dataset_expected", {}).get("train", {}).get("images", 4599))
    train_steps = math.ceil(train_images / int(protocol["batch_size"]))
    warmup_iterations = float(hyp["warmup_epochs"]) * train_steps
    if abs(warmup_iterations - float(protocol["warmup_iterations"])) > 1e-4:
        raise ValueError(
            f"YOLOv5 warmup drift: expected {protocol['warmup_iterations']} iterations, "
            f"got {warmup_iterations}"
        )
    return BenchmarkConfig(config_path, raw)


def ensure_protocol_lock(cfg: BenchmarkConfig, dataset_report: dict) -> dict:
    payload = {
        "schema_version": 1,
        "config_sha256": sha256_file(cfg.path),
        "dataset_yaml_sha256": sha256_file(cfg.dataset_yaml),
        "yolov5_hyp_sha256": sha256_file(cfg.yolov5_hyp),
        "physical_gpu": cfg.raw["physical_gpu"],
        "protocol": cfg.protocol,
        "models": cfg.models,
        "dataset_annotations": {
            split: values["coco_sha256"] for split, values in dataset_report["splits"].items()
        },
        "dataset_content": {
            split: {
                "images_sha256": values["images_sha256"],
                "labels_sha256": values["labels_sha256"],
                "image_list_sha256": values["image_list_sha256"],
            }
            for split, values in dataset_report["splits"].items()
        },
    }
    path = cfg.runtime_root / "protocol.lock.json"
    existing = load_json(path)
    if existing is not None and existing != payload:
        raise RuntimeError("protocol drift detected against runtime/protocol.lock.json")
    atomic_json(path, payload)
    return payload
