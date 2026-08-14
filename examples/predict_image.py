#!/usr/bin/env python3
"""Single-image / folder demo. Dispatches by model_zoo framework."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antiuavdet.registry import get_model, repo_path, config_path  # noqa: E402
from antiuavdet.paths import ROOT as REPO_ROOT  # noqa: E402


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def save_coco(out_dir: Path, image_name: str, boxes_xyxy, scores, thr: float) -> None:
    dets = []
    for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
        if s < thr:
            continue
        dets.append({
            "file_name": image_name,
            "category_id": 0,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(s),
        })
    (out_dir / (Path(image_name).stem + ".json")).write_text(
        json.dumps(dets, indent=2), encoding="utf-8"
    )


def predict_mmdet(args, model_meta):
    from mmdet.apis import init_detector, inference_detector
    cfg = config_path(model_meta, args.dataset)
    det = init_detector(str(cfg), args.weights, device=args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in collect_images(Path(args.source)):
        res = inference_detector(det, str(img))
        b = res.pred_instances.bboxes.cpu().numpy()
        s = res.pred_instances.scores.cpu().numpy()
        save_coco(out, img.name, b, s, args.score_thr)
        print(f"{img.name}: {(s >= args.score_thr).sum()} boxes")


def predict_ultra(args, model_meta):
    import os
    os.environ.setdefault("YOLO_OFFLINE", "true")
    sys.path.insert(0, str(repo_path(model_meta)))
    sys.path.insert(0, str(repo_path(model_meta) / "scripts"))
    if model_meta["id"] == "uavdetr":
        from uavdetr_protocol import disable_online_and_extra_checks, install_fixed_shape
        disable_online_and_extra_checks()
        install_fixed_shape(768, 1344)
        from ultralytics import RTDETR as YOLOCls
        half = False
    elif model_meta["id"] in ("frfdet", "dqef"):
        if model_meta["id"] == "frfdet":
            from frfdet_protocol import disable_online_and_extra_checks
        else:
            from dqef_protocol import disable_online_and_extra_checks
        disable_online_and_extra_checks()
        from ultralytics import YOLO as YOLOCls
        half = True
    else:
        from ultralytics import YOLO as YOLOCls
        half = True
    net = YOLOCls(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in collect_images(Path(args.source)):
        r = net.predict(
            source=str(img), imgsz=(768, 1344), conf=args.score_thr,
            device=args.device.split(":")[-1] if "cuda" in args.device else args.device,
            half=half, verbose=False,
        )[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        s = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        save_coco(out, img.name, b, s, args.score_thr)
        print(f"{img.name}: {len(b)} boxes")


def predict_yaml_family(args, model_meta, engine: str):
    """D-FINE / DEIM / RT-DETR deploy forward."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    repo = str(repo_path(model_meta))
    sys.path.insert(0, repo)
    if engine == "dfine":
        from src.core import YAMLConfig
        normalize = False
    elif engine == "deim":
        from engine.core import YAMLConfig
        normalize = bool(model_meta.get("infer_opts", {}).get("normalize"))
    else:
        from src.core import YAMLConfig
        normalize = False

    cfg = YAMLConfig(str(config_path(model_meta, args.dataset)), resume=args.weights)
    for v in cfg.yaml_cfg.values():
        if isinstance(v, dict) and "pretrained" in v:
            v["pretrained"] = False
    ck = torch.load(args.weights, map_location="cpu")
    state = ck["ema"]["module"] if isinstance(ck, dict) and "ema" in ck else (
        ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    )
    cfg.model.load_state_dict(state)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig):
            return self.postprocessor(self.model(images), orig)

    net = Wrap().to(args.device).eval()
    ops = [T.Resize((768, 1344)), T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tfm = T.Compose(ops)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in collect_images(Path(args.source)):
        pil = Image.open(img).convert("RGB")
        ow, oh = pil.size
        data = tfm(pil).unsqueeze(0).to(args.device)
        orig = torch.tensor([[ow, oh]], device=args.device)
        with torch.no_grad():
            _lab, boxes, scores = net(data, orig)
        b = boxes[0].cpu().numpy()
        s = scores[0].cpu().numpy()
        save_coco(out, img.name, b, s, args.score_thr)
        print(f"{img.name}: {(s >= args.score_thr).sum()} boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--dataset", default="fdd")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "predict"))
    args = ap.parse_args()
    meta = get_model(args.model)
    fw = meta["framework"]
    if fw == "mmdet":
        predict_mmdet(args, meta)
    elif fw == "ultralytics":
        predict_ultra(args, meta)
    elif fw == "dfine":
        predict_yaml_family(args, meta, "dfine")
    elif fw == "deim":
        predict_yaml_family(args, meta, "deim")
    elif fw == "rt-detr":
        predict_yaml_family(args, meta, "rtdetr")
    else:
        raise SystemExit(f"predict not implemented for {fw}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
