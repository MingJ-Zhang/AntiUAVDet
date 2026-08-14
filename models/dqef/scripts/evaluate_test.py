#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image

HEIGHT, WIDTH = 768, 1344


def label_path(image_path: Path, root: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to(root.resolve())
        parts = list(relative.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return (root / Path(*parts)).with_suffix(".txt")
    except ValueError:
        pass
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Cannot map image to label: {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("antiuav", "fdd"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--source-chunk", type=int, default=128)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    os.environ.update({
        "YOLO_CONFIG_DIR": str(project_root / ".ultralytics"),
        "YOLO_OFFLINE": "true",
        "WANDB_DISABLED": "true",
    })
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dqef_protocol import disable_online_and_extra_checks
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    disable_online_and_extra_checks()
    config = yaml.safe_load(
        (project_root / "configs/dataset" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    root = Path(config["path"])
    image_paths: list[Path] = []
    for raw in (root / config["test"]).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            path = Path(raw)
            image_paths.append(path if path.is_absolute() else root / path)

    images, annotations = [], []
    image_ids: dict[str, int] = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        image_ids[str(image_path.resolve())] = image_id
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": original_width,
            "height": original_height,
        })
        current_label = label_path(image_path, root)
        if not current_label.exists():
            continue
        for line in current_label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, x_center, y_center, box_width, box_height = map(float, fields[:5])
            x = (x_center - box_width / 2) * original_width
            y = (y_center - box_height / 2) * original_height
            width = box_width * original_width
            height = box_height * original_height
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

    model = YOLO(str(args.weights))
    predictions = []
    for start in range(0, len(image_paths), args.source_chunk):
        source_paths = image_paths[start:start + args.source_chunk]
        results = model.predict(
            source=[str(path) for path in source_paths],
            imgsz=(HEIGHT, WIDTH),
            conf=0.001,
            iou=0.7,
            max_det=100,
            batch=args.batch,
            device=args.device,
            half=True,
            verbose=False,
        )
        for result in results:
            image_id = image_ids[str(Path(result.path).resolve())]
            for index in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(result.boxes.conf[index]),
                })

    run_dir = project_root / "runs" / f"dqef_t_{args.dataset}_768x1344_300e"
    output_dir = run_dir / "evaluation" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_coco.json"
    predictions_path.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    class_names = config["names"]
    class_name = class_names.get(0, class_names.get("0", "object"))
    ground_truth = {
        "info": {"description": f"{args.dataset} held-out test"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": class_name}],
    }
    gt_path = output_dir / "ground_truth_coco.json"
    gt_path.write_text(json.dumps(ground_truth) + "\n", encoding="utf-8")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))
    evaluation = COCOeval(coco_gt, coco_dt, "bbox")
    evaluation.params.imgIds = [image["id"] for image in images]
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    stats = evaluation.stats

    import torch
    parameter_count = get_num_params(model.model)
    model.model.cpu().float().eval()
    torch.cuda.empty_cache()
    flops = float(get_flops(model.model, imgsz=[HEIGHT, WIDTH]))
    metric_names = ("AP50_95", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR100")
    metric_indices = (0, 1, 2, 3, 4, 5, 8)
    metrics_percent = {
        name: round(float(stats[index]) * 100, 4)
        for name, index in zip(metric_names, metric_indices)
    }
    output = {
        "dataset": args.dataset,
        "split": "test",
        "checkpoint": str(args.weights.resolve()),
        "input_shape_h_w": [HEIGHT, WIDTH],
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 100,
        "images": len(images),
        "ground_truth_boxes": len(annotations),
        "predictions": len(predictions),
        "metrics_percent": metrics_percent,
        "params_m": round(parameter_count / 1e6, 4),
        "flops_g": round(flops, 4),
        "flops_method": "Ultralytics get_flops (THOP multiply-add counted as 2 FLOPs)",
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print("TEST_RESULT_JSON " + json.dumps(output))


if __name__ == "__main__":
    main()
