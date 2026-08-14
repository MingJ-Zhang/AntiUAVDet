#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import torch.multiprocessing as mp
from copy import deepcopy
from pathlib import Path

from frfdet_protocol import disable_online_and_extra_checks, install_fixed_shape

HEIGHT, WIDTH = 768, 1344
DATASETS = ("antiuav", "fdd", "dutdve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dataset", choices=DATASETS, default="antiuav")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    mp.set_sharing_strategy("file_system")
    package_root = Path(__file__).resolve().parents[1]
    os.environ.update({
        "YOLO_CONFIG_DIR": str(project_root / ".ultralytics"),
        "YOLO_OFFLINE": "true",
        "WANDB_DISABLED": "true",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    })

    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    disable_online_and_extra_checks()
    install_fixed_shape(HEIGHT, WIDTH)
    overrides = dict(
        model=str(package_root / "cfg/models/FRFDet/FRFDet-mul.yaml"),
        data=str(project_root / "configs/dataset" / f"{args.dataset}.yaml"),
        project=str(project_root / "runs"),
        name=f"frfdet_t_{args.dataset}_gpu2_preflight",
        exist_ok=True,
        epochs=1,
        batch=args.batch,
        imgsz=WIDTH,
        device="",
        workers=8,
        seed=0,
        deterministic=True,
        pretrained=False,
        amp=args.amp,
        optimizer="AdamW",
        lr0=0.0004,
        lrf=1.0,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=3.0,
        nbs=16,
        fraction=0.01,
        cache=False,
        plots=False,
        val=False,
        save=False,
        rect=False,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        single_cls=True,
        max_det=100,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer._setup_train(world_size=1)
    train_batch = next(iter(trainer.train_loader))
    val_batch = next(iter(trainer.test_loader))
    train_shape = list(train_batch["img"].shape)
    val_shape = list(val_batch["img"].shape)
    if tuple(train_shape[-2:]) != (HEIGHT, WIDTH):
        raise RuntimeError(f"train shape drift: {train_shape}")
    if tuple(val_shape[-2:]) != (HEIGHT, WIDTH):
        raise RuntimeError(f"val shape drift: {val_shape}")

    torch.cuda.reset_peak_memory_stats()
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    train_batch = trainer.preprocess_batch(train_batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        loss, loss_items = trainer.model(train_batch)
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"non-finite loss: {loss}")
    trainer.scaler.scale(loss).backward()
    trainer.optimizer_step()

    trainer.model.eval()
    val_images = val_batch["img"].to(trainer.device, non_blocking=True).float() / 255
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        _ = trainer.model(val_images)
    torch.cuda.synchronize()

    with tempfile.TemporaryDirectory(prefix="frfdet-checkpoint-") as temporary_dir:
        checkpoint_path = Path(temporary_dir) / "roundtrip.pt"
        torch.save({
            "model": deepcopy(trainer.model),
            "ema": deepcopy(trainer.ema.ema),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": 0,
        }, checkpoint_path)
        checkpoint_size = checkpoint_path.stat().st_size
        restored = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_roundtrip = all(restored.get(key) is not None for key in ("model", "ema", "optimizer"))
        if not checkpoint_roundtrip:
            raise RuntimeError("full checkpoint round-trip verification failed")

    result = {
        "ok": True,
        "physical_batch": args.batch,
        "dataset": args.dataset,
        "amp": args.amp,
        "train_shape": train_shape,
        "val_shape": val_shape,
        "loss": float(loss.detach()),
        "loss_items": [float(v) for v in loss_items.detach().flatten()],
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "checkpoint_size_bytes": checkpoint_size,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    output = project_root / "logs" / f"frfdet_t_{args.dataset}_gpu2_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PREFLIGHT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
