#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from copy import copy
from types import MethodType

import torch.multiprocessing as mp

from fdd_benchmark.compat import (
    disable_ultralytics_extra_augmentations,
    disable_yolov13_tensorboard_graph,
    enforce_ultralytics_validation_shape,
    install_ultralytics_fixed_shape,
    prepare_ultralytics_offline_assets,
    register_input_shape_guard,
)
from fdd_benchmark.config import load_config
from fdd_benchmark.io import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
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
            "YOLO_OFFLINE": "true",
            "WANDB_DISABLED": "true",
            "COMET_DISABLE_AUTO_LOGGING": "1",
            "CLEARML_LOG_MODEL": "false",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }
    )
    mp.set_sharing_strategy("file_system")
    prepare_ultralytics_offline_assets(cfg.project_root)
    from ultralytics import YOLO

    disable_ultralytics_extra_augmentations()
    height, width = cfg.input_shape
    is_yolov13 = model_cfg["backend"] == "yolov13"
    install_ultralytics_fixed_shape(height, width, yolov13=is_yolov13)

    run_dir = cfg.runs_root / args.model
    predictions_dir = run_dir / "val_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "train" / "weights" / "last.pt"
    model_path = last if args.resume else cfg.project_root / model_cfg["weight"]
    if args.resume and not last.exists():
        raise FileNotFoundError(last)
    model = YOLO(str(model_path))
    if is_yolov13:
        disable_yolov13_tensorboard_graph(model)

    phase_seen: set[int] = set()

    def enforce_strict_numerics(trainer) -> None:
        enforce_ultralytics_validation_shape(trainer, height, width)
        register_input_shape_guard(
            trainer,
            height,
            width,
            lambda shape: atomic_json(
                run_dir / "input_shape_guard.json",
                {"expected_h_w": [height, width], "first_batch_shape": list(shape)},
            ),
        )

        def fail_instead_of_recover(self, _epoch: int) -> bool:
            loss_bad = self.loss is not None and not bool(self.loss.isfinite())
            fitness_bad = self.fitness is not None and not math.isfinite(float(self.fitness))
            collapse = bool(
                self.best_fitness
                and self.best_fitness > 0
                and self.fitness is not None
                and float(self.fitness) == 0.0
            )
            if loss_bad or fitness_bad or collapse:
                reason = (
                    "loss NaN/Inf"
                    if loss_bad
                    else "fitness NaN/Inf"
                    if fitness_bad
                    else "fitness collapse"
                )
                raise RuntimeError(f"protocol stop: {reason}; automatic recovery is disabled")
            return False

        if hasattr(trainer, "_handle_nan_recovery"):
            trainer._handle_nan_recovery = MethodType(fail_instead_of_recover, trainer)

    def apply_phase(trainer) -> None:
        # 新版 Ultralytics 首轮会自动减半 batch；正式协议要求 OOM 直接失败。
        if hasattr(trainer, "_oom_retries"):
            trainer._oom_retries = 3
        epoch = int(trainer.epoch)
        phase = (
            0
            if epoch < a["active_start_epoch"]
            else 1
            if epoch < a["active_stop_epoch"]
            else 2
            if epoch < a["no_aug_start_epoch"]
            else 3
        )
        if phase in phase_seen:
            return
        phase_seen.add(phase)
        trainer.args.mosaic = a["mosaic"] if phase == 1 else 0.0
        trainer.args.mixup = a["mixup"] if phase == 1 else 0.0
        if phase == 3:
            for name in (
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
            ):
                setattr(trainer.args, name, 0.0)
        dataset = trainer.train_loader.dataset
        if hasattr(dataset, "mosaic"):
            dataset.mosaic = phase == 1
        if hasattr(dataset, "build_transforms"):
            dataset.transforms = dataset.build_transforms(hyp=copy(trainer.args))
        if hasattr(trainer.train_loader, "reset"):
            trainer.train_loader.reset()
        atomic_json(
            run_dir / "augmentation_phase.json",
            {
                "epoch": epoch,
                "phase": phase,
                "mosaic": trainer.args.mosaic,
                "mixup": trainer.args.mixup,
            },
        )

    def save_predictions(trainer) -> None:
        # final_eval() 会把 native best 再验证一次并临时令 epoch=epochs；该结果不属于
        # 新训练轮次，必须排除，否则会污染中央 best-checkpoint 选择。
        if int(trainer.epoch) >= int(cfg.protocol["epochs"]):
            return
        jdict = getattr(getattr(trainer, "validator", None), "jdict", None)
        if jdict is None:
            return
        atomic_json(predictions_dir / f"epoch_{int(trainer.epoch):03d}.raw.json", jdict)

    model.add_callback("on_train_start", enforce_strict_numerics)
    model.add_callback("on_train_epoch_start", apply_phase)
    model.add_callback("on_fit_epoch_end", save_predictions)
    p = cfg.protocol
    a = p["augmentation"]
    train_list = (cfg.dataset_view / "lists" / "train.txt").read_text().splitlines()
    train_images = len(
        [line for line in train_list if line]
    )
    steps = math.ceil(train_images / p["batch_size"])
    train_args = dict(
        data=str(cfg.dataset_yaml),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        epochs=p["epochs"],
        batch=p["batch_size"],
        # 训练器保留长边标量，数据变换工厂负责输出固定 768×1344。
        imgsz=cfg.input_long_side,
        # 空字符串表示可见集合内的 cuda:0，避免旧后端覆盖外部物理卡绑定。
        device="",
        workers=p["workers"],
        seed=p["seed"],
        deterministic=p["deterministic"],
        amp=p["amp"],
        optimizer=p["optimizer"],
        lr0=p["learning_rate"],
        lrf=p["final_lr_fraction"],
        momentum=p["beta1"],
        weight_decay=p["weight_decay"],
        warmup_epochs=p["warmup_iterations"] / steps,
        warmup_momentum=p["warmup_momentum"],
        warmup_bias_lr=p["warmup_bias_lr"],
        patience=p["patience"],
        save=True,
        save_period=p["save_period"],
        val=True,
        plots=True,
        save_json=True,
        conf=p["conf_threshold_eval"],
        iou=p["nms_iou_threshold"],
        max_det=p["max_detections"],
        single_cls=True,
        # v8/v11 将 multi_scale 定义为浮点比例，v13 官方版本仍要求布尔值。
        multi_scale=False if is_yolov13 else 0.0,
        close_mosaic=12,
        nbs=16,
        cache=False,
        rect=False,
        hsv_h=a["hsv_h"],
        hsv_s=a["hsv_s"],
        hsv_v=a["hsv_v"],
        degrees=a["degrees"],
        translate=a["translate"],
        scale=a["scale"],
        shear=a["shear"],
        perspective=a["perspective"],
        flipud=a["flipud"],
        fliplr=a["fliplr"],
        mosaic=a["mosaic"],
        mixup=a["mixup"],
        copy_paste=0.0,
        verbose=True,
    )
    if args.resume:
        model.train(resume=True)
    else:
        atomic_json(run_dir / "resolved_train_args.json", train_args)
        model.train(**train_args)


if __name__ == "__main__":
    main()
