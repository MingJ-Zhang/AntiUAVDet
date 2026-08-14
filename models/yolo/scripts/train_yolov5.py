#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch.multiprocessing as mp

from fdd_benchmark.compat import (
    disable_yolov5_extra_augmentations,
    enable_pillow_legacy_getsize,
)
from fdd_benchmark.config import load_config
from fdd_benchmark.io import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="yolov5n")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]
    source = cfg.project_root / model_cfg["source"]
    sys.path.insert(0, str(source))
    os.environ.update(
        {
            "YOLO_CONFIG_DIR": str(cfg.project_root / ".ultralytics"),
            "YOLOV5_CONFIG_DIR": str(cfg.project_root / "vendor/fonts"),
            "YOLO_OFFLINE": "true",
            "WANDB_DISABLED": "true",
        }
    )
    mp.set_sharing_strategy("file_system")
    enable_pillow_legacy_getsize()
    import train
    from utils.callbacks import Callbacks

    disable_yolov5_extra_augmentations()

    run_dir = cfg.runs_root / args.model
    predictions = run_dir / "val_predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    native_run = run_dir / "train"
    raw_prediction = native_run / "_predictions.json"
    callbacks = Callbacks()
    shape_recorded = False

    def save_predictions(_log_values, epoch, _best_fitness, _fitness) -> None:
        if raw_prediction.exists():
            shutil.copy2(raw_prediction, predictions / f"epoch_{int(epoch):03d}.raw.json")

    callbacks.register_action("on_fit_epoch_end", "save_fdd_predictions", save_predictions)
    height, width = cfg.input_shape

    def save_input_shape(_model, _ni, images, _targets, _paths, _losses) -> None:
        nonlocal shape_recorded
        if shape_recorded:
            return
        shape_recorded = True
        atomic_json(
            run_dir / "input_shape_guard.json",
            {
                "expected_h_w": [height, width],
                "first_batch_shape": list(images.shape),
            },
        )

    callbacks.register_action("on_train_batch_end", "save_fdd_input_shape", save_input_shape)
    p = cfg.protocol
    argv = [
        "train.py",
        "--weights",
        str(cfg.project_root / model_cfg["weight"]),
        "--data",
        str(cfg.dataset_yaml),
        "--hyp",
        str(cfg.yolov5_hyp),
        "--epochs",
        str(p["epochs"]),
        "--batch-size",
        str(p["batch_size"]),
        "--imgsz",
        str(cfg.input_long_side),
        "--img-height",
        str(height),
        "--img-width",
        str(width),
        "--workers",
        str(p["workers"]),
        "--optimizer",
        "AdamW",
        "--patience",
        str(p["patience"]),
        "--save-period",
        "1",
        "--seed",
        str(p["seed"]),
        "--single-cls",
        "--project",
        str(run_dir),
        "--name",
        "train",
        "--exist-ok",
    ]
    last = native_run / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise FileNotFoundError(last)
        argv = ["train.py", "--resume", str(last)]
    sys.argv = argv
    old_cwd = Path.cwd()
    os.chdir(source)
    try:
        train.main(train.parse_opt(), callbacks)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
