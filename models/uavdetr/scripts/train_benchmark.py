#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import torch.multiprocessing as mp
from copy import copy
from pathlib import Path

from uavdetr_protocol import disable_online_and_extra_checks, install_fixed_shape, register_shape_guard

HEIGHT, WIDTH = 768, 1344
PHYSICAL_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 16
DATASET_EPOCHS = {'antiuav': 300, 'fdd': 300, 'dut_dve': 200}
DATASETS = tuple(DATASET_EPOCHS)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    socket.setdefaulttimeout(15)
    mp.set_sharing_strategy('file_system')
    repo = Path(__file__).resolve().parents[1]
    project_root = args.project_root.resolve()
    epochs = args.epochs or DATASET_EPOCHS[args.dataset]
    run_name = f'uavdetr_{args.dataset}_768x1344_{epochs}e'
    run_dir = project_root / 'runs' / run_name
    last = run_dir / 'weights' / 'last.pt'

    os.environ.update({
        'YOLO_CONFIG_DIR': str(project_root / '.ultralytics'),
        'YOLO_OFFLINE': 'true',
        'WANDB_DISABLED': 'true',
        'COMET_DISABLE_AUTO_LOGGING': '1',
        'CLEARML_LOG_MODEL': 'false',
        'NO_ALBUMENTATIONS_UPDATE': '1',
        'ULTRALYTICS_AMP_DTYPE': 'bfloat16',
    })

    from ultralytics import RTDETR

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
            trainer, HEIGHT, WIDTH,
            lambda shape: write_json(
                run_dir / 'input_shape_guard.json',
                {'expected_h_w': [HEIGHT, WIDTH], 'first_batch_shape': list(shape)},
            ),
        )

    def apply_phase(trainer) -> None:
        epoch = int(trainer.epoch)
        phase = 0 if epoch < 4 else 1 if epoch < 78 else 2 if epoch < epochs - 12 else 3
        if phase in phase_seen:
            return
        phase_seen.add(phase)
        trainer.args.mosaic = 0.5 if phase == 1 else 0.0
        trainer.args.mixup = 0.5 if phase == 1 else 0.0
        if phase == 3:
            for name in ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                         'shear', 'perspective', 'flipud', 'fliplr'):
                setattr(trainer.args, name, 0.0)
        dataset = trainer.train_loader.dataset
        dataset.transforms = dataset.build_transforms(hyp=copy(trainer.args))
        if hasattr(trainer.train_loader, 'reset'):
            trainer.train_loader.reset()
        write_json(run_dir / 'augmentation_phase.json', {
            'epoch': epoch,
            'phase': phase,
            'mosaic': trainer.args.mosaic,
            'mixup': trainer.args.mixup,
        })

    model = RTDETR(str(last) if args.resume else str(repo / 'ultralytics/cfg/models/UAV-DETR.yaml'))
    model.add_callback('on_train_batch_start', before_first_batch)
    model.add_callback('on_train_epoch_start', apply_phase)

    train_args = dict(
        data=str(project_root / 'configs/dataset' / f'{args.dataset}.yaml'),
        project=str(project_root / 'runs'),
        name=run_name,
        exist_ok=True,
        epochs=epochs,
        batch=PHYSICAL_BATCH_SIZE,
        imgsz=WIDTH,
        device='',
        workers=8,
        seed=0,
        deterministic=True,
        pretrained=False,
        amp=True,
        optimizer='AdamW',
        lr0=0.0004,
        lrf=1.0,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=1000,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=epochs,
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
        model.train(resume=str(last), imgsz=WIDTH, batch=PHYSICAL_BATCH_SIZE)
    else:
        write_json(run_dir / 'resolved_train_args.json', train_args)
        model.train(**train_args)


if __name__ == '__main__':
    main()
