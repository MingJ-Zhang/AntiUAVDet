#!/usr/bin/env python3
"""RT-DETR official-repo test-set inference → raw COCO detections.

Preprocess: Resize(1344, 768) + /255, no ImageNet normalize.
Postprocessor maps boxes back with orig_target_sizes=[W, H].
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from src.core import YAMLConfig

    gt = json.load(open(a.gt, encoding="utf-8"))
    cfg = YAMLConfig(a.config, resume=a.checkpoint)
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy().to(a.device).eval()
    postprocessor = cfg.postprocessor.deploy()

    preds = []
    for im in gt["images"]:
        fid = im["file_name"]
        iid = im["id"]
        W, H = im["width"], im["height"]
        p = os.path.join(a.img_dir, fid)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB").resize((1344, 768), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        inp = t.unsqueeze(0).to(a.device)
        orig = torch.tensor([[W, H]], device=a.device)
        with torch.no_grad():
            labels, boxes, scores = postprocessor(model(inp), orig)
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        for bx, s in zip(boxes, scores):
            if s < a.score_thr:
                continue
            x1, y1, x2, y2 = bx.tolist()
            preds.append({
                "image_id": iid,
                "category_id": 0,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    os.makedirs(os.path.dirname(a.out_raw) or ".", exist_ok=True)
    with open(a.out_raw, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[rtdetr] {a.model}/{a.dataset}: {len(preds)} raw dets "
          f"over {len(gt['images'])} images")


if __name__ == "__main__":
    main()
