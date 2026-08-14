#!/usr/bin/env python3
"""Validate the unified prediction schema. Must print RESULT: PASS."""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("gt")
    ap.add_argument("--max-bad", type=int, default=20)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    pr = json.load(open(args.pred, encoding="utf-8"))

    gt_imgs = {im["id"]: im for im in gt["images"]}
    gt_fname_to_id = {im["file_name"]: im["id"] for im in gt["images"]}

    if isinstance(pr, list):
        preds = pr
        meta = {}
    else:
        meta = pr.get("meta", {})
        preds = pr.get("predictions", [])

    errors = []
    per_img = {}
    score_min, score_max = 1.0, 0.0

    for idx, p in enumerate(preds):
        iid = p.get("image_id")
        if iid not in gt_imgs:
            errors.append(f"[{idx}] image_id {iid} not in GT")
            continue
        im = gt_imgs[iid]
        w, h = im["width"], im["height"]
        fname = p.get("file_name")
        if fname and gt_fname_to_id.get(fname) != iid:
            errors.append(f"[{idx}] file_name {fname} does not match image_id {iid}")

        bbox = p.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"[{idx}] bad bbox length {len(bbox)}")
            continue
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            errors.append(f"[{idx}] non-positive w/h: {bbox}")
        if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
            errors.append(f"[{idx}] bbox out of range {bbox} vs {w}x{h}")

        score = p.get("score", 0)
        if not (0.0 <= score <= 1.0):
            errors.append(f"[{idx}] score out of [0,1]: {score}")
        score_min = min(score_min, score)
        score_max = max(score_max, score)

        if p.get("category_id", 0) != 0:
            errors.append(f"[{idx}] category_id != 0: {p.get('category_id')}")

        per_img[iid] = per_img.get(iid, 0) + 1

    missing = [iid for iid in gt_imgs if iid not in per_img]
    print(f"GT images: {len(gt_imgs)}")
    print(f"Predictions: {len(preds)}")
    print(f"Images with >=1 pred: {len(per_img)}")
    print(f"Images with 0 pred: {len(missing)}")
    print(f"Score range: {score_min:.4f} ~ {score_max:.4f}")
    if meta:
        print(f"Meta: {json.dumps(meta, ensure_ascii=False)}")
    if errors:
        print(f"ERRORS ({len(errors)}, show first {args.max_bad}):")
        for e in errors[:args.max_bad]:
            print("  ", e)
    ok = len(errors) == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
