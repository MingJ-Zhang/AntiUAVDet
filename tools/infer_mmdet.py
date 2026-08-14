#!/usr/bin/env python3
"""MMDetection test-set inference → raw COCO detections (original-image xywh)."""
import argparse
import json
import os

from mmdet.apis import init_detector, inference_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--max-dets", type=int, default=300)
    a = ap.parse_args()

    gt = json.load(open(a.gt, encoding="utf-8"))
    model = init_detector(a.config, a.checkpoint, device=a.device)

    preds = []
    skipped = 0
    for im in gt["images"]:
        p = os.path.join(a.img_dir, im["file_name"])
        if not os.path.exists(p):
            skipped += 1
            continue
        res = inference_detector(model, p)
        bboxes = res.pred_instances.bboxes.cpu().numpy()
        scores = res.pred_instances.scores.cpu().numpy()
        kept = 0
        for (x1, y1, x2, y2), s in zip(bboxes, scores):
            if s < a.score_thr:
                continue
            preds.append({
                "image_id": im["id"],
                "category_id": 0,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
            kept += 1
            if kept >= a.max_dets:
                break

    os.makedirs(os.path.dirname(a.out_raw) or ".", exist_ok=True)
    with open(a.out_raw, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[mmdet] {a.model}/{a.dataset}: {len(preds)} raw dets "
          f"over {len(gt['images'])} images, skipped={skipped}")


if __name__ == "__main__":
    main()
