#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

from fdd_benchmark.compat import (
    enable_pillow_legacy_getsize,
    prepare_ultralytics_offline_assets,
)
from fdd_benchmark.config import load_config
from fdd_benchmark.io import atomic_json
from fdd_benchmark.metrics import evaluate_prediction_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
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
        }
    )
    prepare_ultralytics_offline_assets(cfg.project_root)
    run_dir = cfg.runs_root / args.model
    selected = cfg.artifacts_root / "selected" / f"{args.model}.pt"
    if not selected.exists():
        raise FileNotFoundError(selected)
    eval_dir = run_dir / "evaluation" / args.split
    completed_metrics = eval_dir / "metrics.json"
    if args.split == "test" and completed_metrics.exists():
        print(f"test evaluation already completed; reusing {completed_metrics}")
        return
    p = cfg.protocol
    height, width = cfg.input_shape
    annotation = json.loads(cfg.annotation(args.split).read_text(encoding="utf-8"))
    image_paths = [
        cfg.image_dir(args.split) / item["file_name"]
        for item in annotation["images"]
    ]
    predictions: list[dict] = []
    if model_cfg["backend"] == "yolov5":
        enable_pillow_legacy_getsize()
        from models.experimental import attempt_load
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        device = torch.device("cuda:0")
        model = attempt_load(str(selected), device=device).eval().half()
        for start in range(0, len(image_paths), p["batch_size"]):
            paths = image_paths[start : start + p["batch_size"]]
            originals = [cv2.imread(str(path)) for path in paths]
            for path, image in zip(paths, originals, strict=True):
                if image is None:
                    raise FileNotFoundError(path)
            prepared = [
                letterbox(image, new_shape=(height, width), auto=False, stride=32)[0]
                for image in originals
            ]
            tensor = np.stack(
                [np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)) for image in prepared]
            )
            tensor = torch.from_numpy(tensor).to(device).half() / 255.0
            if tuple(tensor.shape[-2:]) != (height, width):
                raise RuntimeError(f"inference input-shape drift: {tuple(tensor.shape)}")
            with torch.inference_mode():
                outputs = model(tensor)
                detections = non_max_suppression(
                    outputs[0] if isinstance(outputs, tuple | list) else outputs,
                    conf_thres=p["conf_threshold_eval"],
                    iou_thres=p["nms_iou_threshold"],
                    max_det=p["max_detections"],
                )
            for path, original, detected in zip(paths, originals, detections, strict=True):
                # NMS 输出继承 inference tensor 属性；坐标恢复会执行原地运算，需先克隆。
                detected = detected.clone()
                if len(detected):
                    detected[:, :4] = scale_boxes(
                        tensor.shape[2:], detected[:, :4], original.shape
                    ).round()
                for x1, y1, x2, y2, score, _cls in detected.tolist():
                    predictions.append(
                        {
                            "image_id": path.stem,
                            "category_id": 0,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
    else:
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionPredictor

        class ProtocolPredictor(DetectionPredictor):
            """在真实推理张量进入模型前执行尺寸、batch 和精度门禁。"""

            shape_recorded = False

            def preprocess(self, images):
                tensor = super().preprocess(images)
                actual = tuple(int(value) for value in tensor.shape)
                if actual[-2:] != (height, width):
                    raise RuntimeError(
                        f"inference input-shape drift: expected (*,3,{height},{width}), "
                        f"got {actual}"
                    )
                if actual[0] > p["batch_size"]:
                    raise RuntimeError(f"inference batch drift: got {actual[0]}")
                if tensor.dtype != torch.float16:
                    raise RuntimeError(f"inference precision drift: got {tensor.dtype}")
                if not self.shape_recorded:
                    self.shape_recorded = True
                    atomic_json(
                        eval_dir / "input_shape_guard.json",
                        {
                            "expected_h_w": [height, width],
                            "first_batch_shape": list(actual),
                            "dtype": str(tensor.dtype),
                        },
                    )
                return tensor

        model = YOLO(str(selected))
        precision_args = (
            {"half": True}
            if model_cfg["backend"] == "yolov13"
            else {"quantize": 16}
        )
        # 上游把路径 list 视为内存输入，并将 list 长度直接作为 batch；因此必须显式
        # 按协议 batch 分块，不能一次传入整个 split。
        for start in range(0, len(image_paths), p["batch_size"]):
            paths = image_paths[start : start + p["batch_size"]]
            results = model.predict(
                source=[str(path) for path in paths],
                imgsz=(height, width),
                batch=p["batch_size"],
                conf=p["conf_threshold_eval"],
                iou=p["nms_iou_threshold"],
                max_det=p["max_detections"],
                device="",
                rect=False,
                stream=True,
                verbose=False,
                predictor=ProtocolPredictor,
                **precision_args,
            )
            for result, path in zip(results, paths, strict=True):
                if tuple(result.boxes.orig_shape) != tuple(result.orig_shape):
                    raise RuntimeError("prediction coordinate restoration drift")
                for box, score in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    x1, y1, x2, y2 = box
                    predictions.append(
                        {
                            "image_id": path.stem,
                            "category_id": 0,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
    raw = eval_dir / "predictions.raw.json"
    atomic_json(raw, predictions)
    evaluate_prediction_file(
        raw,
        cfg.annotation(args.split),
        eval_dir / "predictions.coco.json",
        eval_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
