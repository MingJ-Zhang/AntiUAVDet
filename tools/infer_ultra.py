#!/usr/bin/env python3
"""UAV-DETR test-set inference → raw COCO detections (original-image xywh).

Protocol (required for correctness):
  fp32 + install_fixed_shape(768, 1344). Mixed-precision WTConv produces
  giant boxes and AP ≈ 0.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="uavdetr", choices=["uavdetr"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.repo)))
    sys.path.insert(0, str(Path(a.repo) / "scripts"))
    os.environ.update({
        "YOLO_OFFLINE": "true",
        "WANDB_DISABLED": "true",
        "COMET_DISABLE_AUTO_LOGGING": "1",
    })

    from uavdetr_protocol import disable_online_and_extra_checks, install_fixed_shape
    disable_online_and_extra_checks()
    install_fixed_shape(768, 1344)

    import torch
    from ultralytics import RTDETR
    import ultralytics.nn.tasks as _tasks

    if a.device != "cpu":
        torch.cuda.init()
    _tasks.DetectionModel.fuse = lambda self, verbose=True: self

    gt = json.load(open(a.gt, encoding="utf-8"))
    model = RTDETR(str(a.weights))

    preds = []
    imgs = gt["images"]
    for start in range(0, len(imgs), a.chunk):
        sub = imgs[start:start + a.chunk]
        paths = [os.path.join(a.img_dir, im["file_name"]) for im in sub]
        results = model.predict(
            source=paths,
            imgsz=(768, 1344),
            conf=a.score_thr,
            iou=0.7,
            max_det=100,
            batch=a.batch,
            device=a.device,
            half=False,
            verbose=False,
        )
        for im, r in zip(sub, results):
            boxes = r.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                s = float(boxes.conf[i])
                if s < a.score_thr:
                    continue
                preds.append({
                    "image_id": im["id"],
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": s,
                })

    os.makedirs(os.path.dirname(a.out_raw) or ".", exist_ok=True)
    with open(a.out_raw, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[{a.kind}] {a.model}/{a.dataset}: {len(preds)} raw dets "
          f"over {len(imgs)} images")


if __name__ == "__main__":
    main()
