#!/usr/bin/env python3
"""Wrap a raw detection list into the unified AntiUAVDet prediction schema."""
import argparse
import datetime
import json
import os
from collections import defaultdict

MAX_DETS = 300
SCORE_THR = 0.001


def load_gt(path):
    gt = json.load(open(path, encoding="utf-8"))
    by_id = {im["id"]: im for im in gt["images"]}
    by_name = {im["file_name"]: im["id"] for im in gt["images"]}
    by_stem = {os.path.splitext(im["file_name"])[0]: im["id"] for im in gt["images"]}
    images = [
        {"id": im["id"], "file_name": im["file_name"],
         "width": im["width"], "height": im["height"]}
        for im in gt["images"]
    ]
    return by_id, by_name, by_stem, images


def resolve_id(raw_id, by_id, by_name, by_stem):
    if isinstance(raw_id, int) and raw_id in by_id:
        return raw_id
    s = str(raw_id)
    if s in by_name:
        return by_name[s]
    stem = os.path.splitext(os.path.basename(s))[0]
    if stem in by_stem:
        return by_stem[stem]
    if s.isdigit() and int(s) in by_id:
        return int(s)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--framework", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--epoch", default=None)
    ap.add_argument("--id-mode", choices=("auto", "gt", "name"), default="auto")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR)
    ap.add_argument("--max-dets", type=int, default=MAX_DETS)
    ap.add_argument("--clip", action="store_true", default=True)
    ap.add_argument("--extra-meta", default="{}")
    args = ap.parse_args()

    by_id, by_name, by_stem, images = load_gt(args.gt)
    raw = json.load(open(args.raw, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("predictions", raw.get("annotations", []))

    per_img = defaultdict(list)
    unresolved = 0
    for d in raw:
        iid = resolve_id(d["image_id"], by_id, by_name, by_stem)
        if iid is None:
            unresolved += 1
            continue
        score = float(d["score"])
        if score < args.score_thr:
            continue
        x, y, w, h = [float(v) for v in d["bbox"]]
        if args.clip:
            im = by_id[iid]
            W, H = im["width"], im["height"]
            x2, y2 = x + w, y + h
            x, y = max(0.0, min(x, W)), max(0.0, min(y, H))
            x2, y2 = max(0.0, min(x2, W)), max(0.0, min(y2, H))
            w, h = x2 - x, y2 - y
        if w <= 0 or h <= 0:
            continue
        per_img[iid].append({
            "image_id": iid,
            "file_name": by_id[iid]["file_name"],
            "category_id": 0,
            "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
            "score": round(score, 6),
            "area": round(w * h, 3),
            "iscrowd": 0,
        })

    preds = []
    for iid in sorted(per_img):
        items = sorted(per_img[iid], key=lambda d: -d["score"])[: args.max_dets]
        preds.extend(items)

    epoch = args.epoch
    if epoch is not None:
        try:
            epoch = int(epoch)
        except ValueError:
            pass

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "framework": args.framework,
        "img_size": [768, 1344],
        "score_thr_stored": args.score_thr,
        "max_dets_per_image": args.max_dets,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_images": len(images),
        "num_predictions": len(preds),
        "images_with_pred": len(per_img),
        "unresolved_image_ids": unresolved,
    }
    meta.update(json.loads(args.extra_meta))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "images": images, "predictions": preds},
                  f, ensure_ascii=False)
    with open(args.out.replace(".json", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[wrap] {args.model}/{args.dataset}: {len(preds)} dets over "
          f"{len(per_img)}/{len(images)} images, unresolved={unresolved}")


if __name__ == "__main__":
    main()
