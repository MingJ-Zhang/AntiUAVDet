#!/usr/bin/env python3
"""FRFDet / DQEF-Net test-set inference → raw COCO detections."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["frfdet", "dqef"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--half", action="store_true", default=False)
    ap.add_argument("--imgsz", default="768,1344")
    a = ap.parse_args()

    repo = Path(a.repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    os.environ.update({
        "YOLO_OFFLINE": "true",
        "WANDB_DISABLED": "true",
        "COMET_DISABLE_AUTO_LOGGING": "1",
    })

    if a.kind == "frfdet":
        from frfdet_protocol import disable_online_and_extra_checks
    else:
        from dqef_protocol import disable_online_and_extra_checks
    disable_online_and_extra_checks()

    if a.device != "cpu":
        torch.cuda.init()
    from ultralytics import YOLO
    import ultralytics.nn.tasks as _tasks
    _tasks.DetectionModel.fuse = lambda self, verbose=True: self
    try:
        from ultralytics.utils import checks as _checks
        import ultralytics.data.utils as _data_utils
        _checks.check_font = lambda *_a, **_k: None
        _data_utils.check_font = lambda *_a, **_k: None
    except Exception:
        pass

    h, w = map(int, a.imgsz.split(","))
    gt = json.load(open(a.gt, encoding="utf-8"))
    model = YOLO(str(a.weights))

    preds = []
    imgs = gt["images"]
    chunk = max(1, min(a.batch, 32))
    for start in range(0, len(imgs), chunk):
        sub = imgs[start:start + chunk]
        paths = [os.path.join(a.img_dir, im["file_name"]) for im in sub]
        results = model.predict(
            source=paths,
            imgsz=(h, w),
            conf=a.score_thr,
            iou=0.7,
            max_det=a.max_det,
            batch=a.batch,
            device=a.device,
            half=a.half,
            verbose=False,
        )
        for im, r in zip(sub, results):
            bx = r.boxes
            for i in range(len(bx)):
                x1, y1, x2, y2 = bx.xyxy[i].tolist()
                s = float(bx.conf[i])
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
