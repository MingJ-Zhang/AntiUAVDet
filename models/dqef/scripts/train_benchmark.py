#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import torch.multiprocessing as mp
from copy import copy
from pathlib import Path

from dqef_protocol import disable_online_and_extra_checks, install_fixed_shape, register_shape_guard

HEIGHT, WIDTH = 768, 1344
EPOCHS = 300
PHYSICAL_BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 16
DATASETS = ("antiuav", "fdd", "dutdve")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=PHYSICAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    project_root = args.project_root.resolve()
    run_name = f"dqef_t_{args.dataset}_768x1344_{args.epochs}e"
    run_dir = project_root / "runs" / run_name
    last = run_dir / "weights" / "last.pt"

    os.environ.update({
        "YOLO_CONFIG_DIR": str(project_root / ".ultralytics"),
        "YOLO_OFFLINE": "true",
        "WANDB_DISABLED": "true",
        "COMET_DISABLE_AUTO_LOGGING": "1",
        "CLEARML_LOG_MODEL": "false",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    mp.set_sharing_strategy("file_system")

    from ultralytics import YOLO

    disable_online_and_extra_checks()
    install_fixed_shape(HEIGHT, WIDTH)
    phase_seen: set[int] = set()
    guard_registered = False

    def before_first_batch(trainer) -> None:
        nonlocal guard_registered
        if guard_registered:
            return
        guard_registered = True
        register_shape_guard(
            trainer,
            HEIGHT,
            WIDTH,
            lambda shape: write_json(
                run_dir / "input_shape_guard.json",
                {"expected_h_w": [HEIGHT, WIDTH], "first_batch_shape": list(shape)},
            ),
        )

    def apply_phase(trainer) -> None:
        epoch = int(trainer.epoch)
        phase = 0 if epoch < 4 else 1 if epoch < 78 else 2 if epoch < 288 else 3
        if phase in phase_seen:
            return
        phase_seen.add(phase)
        trainer.args.mosaic = 0.5 if phase == 1 else 0.0
        trainer.args.mixup = 0.5 if phase == 1 else 0.0
        if phase == 3:
            for name in (
                "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "flipud", "fliplr",
            ):
                setattr(trainer.args, name, 0.0)
        dataset = trainer.train_loader.dataset
        dataset.transforms = dataset.build_transforms(hyp=copy(trainer.args))
        if hasattr(trainer.train_loader, "reset"):
            trainer.train_loader.reset()
        write_json(run_dir / "augmentation_phase.json", {
            "epoch": epoch,
            "phase": phase,
            "mosaic": trainer.args.mosaic,
            "mixup": trainer.args.mixup,
        })

    model = YOLO(str(last) if args.resume else str(repo / "ultralytics/cfg/models/DQCF-Net-T.yaml"))
    model.add_callback("on_train_batch_start", before_first_batch)
    model.add_callback("on_train_epoch_start", apply_phase)
    train_args = dict(
        data=str(project_root / "configs/dataset" / f"{args.dataset}.yaml"),
        project=str(project_root / "runs"),
        name=run_name,
        exist_ok=True,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=WIDTH,
        device="",
        workers=8,
        seed=0,
        deterministic=True,
        pretrained=False,
        amp=True,
        optimizer="AdamW",
        lr0=0.0004,
        lrf=1.0,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=args.epochs,
        save=True,
        save_period=-1,
        val=True,
        plots=False,
        save_json=False,
        conf=0.001,
        iou=0.7,
        max_det=100,
        single_cls=True,
        close_mosaic=12,
        nbs=EFFECTIVE_BATCH_SIZE,
        fraction=1.0,
        cache=False,
        rect=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.5,
        copy_paste=0.0,
        verbose=True,
    )
    if args.resume:
        model.train(resume=str(last), imgsz=WIDTH, batch=args.batch, workers=8)
    else:
        write_json(run_dir / "resolved_train_args.json", train_args)
        model.train(**train_args)


if __name__ == "__main__":
    main()
