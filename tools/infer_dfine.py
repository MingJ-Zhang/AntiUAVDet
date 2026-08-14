#!/usr/bin/env python3
"""D-FINE test-set inference → raw COCO detections.

Resize(eval_spatial_size) + ToTensor, no ImageNet mean/std (HGNetv2-N).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
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
    ap.add_argument("--size", default="768,1344")
    ap.add_argument("--score-thr", type=float, default=0.001)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from src.core import YAMLConfig

    cfg = YAMLConfig(a.config, resume=a.checkpoint)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(a.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = a.device
    model = Model().to(device).eval()
    h, w = map(int, a.size.split(","))
    resize = T.Resize((h, w))
    to_tensor = T.ToTensor()

    gt = json.load(open(a.gt, encoding="utf-8"))
    preds = []
    skipped = 0
    for im in gt["images"]:
        p = os.path.join(a.img_dir, im["file_name"])
        if not os.path.exists(p):
            skipped += 1
            continue
        pil = Image.open(p).convert("RGB")
        ow, oh = pil.size
        orig = torch.tensor([[ow, oh]]).to(device)
        data = to_tensor(resize(pil)).unsqueeze(0).to(device)
        with torch.no_grad():
            _labels, boxes, scores = model(data, orig)
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s < a.score_thr:
                continue
            preds.append({
                "image_id": im["id"],
                "category_id": 0,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    os.makedirs(os.path.dirname(a.out_raw) or ".", exist_ok=True)
    with open(a.out_raw, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[dfine] {a.model}/{a.dataset}: {len(preds)} raw dets "
          f"over {len(gt['images'])} images, skipped={skipped}")


if __name__ == "__main__":
    main()
