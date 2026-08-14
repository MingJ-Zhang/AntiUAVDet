#!/usr/bin/env python3
"""COCO 7 metrics + best-F1 / PR curve."""
import argparse
import datetime
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco7(ev):
    s = ev.stats
    keys = ["AP50:95", "AP50", "AP75", "AP-s", "AP-m", "AP-l", "AR100"]
    idx = [0, 1, 2, 3, 4, 5, 8]
    return {k: round(float(s[i]), 4) for k, i in zip(keys, idx)}


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix, iy = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def best_f1(cocoGt, dets, thr_lo=0.01, thr_hi=0.95, step=0.01):
    from collections import defaultdict

    gtim = defaultdict(list)
    for a in cocoGt.loadAnns(cocoGt.getAnnIds()):
        gtim[a["image_id"]].append(a)
    detim = defaultdict(list)
    for d in dets:
        detim[d["image_id"]].append(d)

    thrs = np.arange(thr_lo, thr_hi + 1e-9, step)
    rec, prec, f1s = [], [], []
    for t in thrs:
        tp = fp = fn = 0
        for iid in gtim:
            gs = gtim[iid]
            ds = sorted(detim.get(iid, []), key=lambda d: -d["score"])
            used = [False] * len(gs)
            for d in ds:
                if d["score"] < t:
                    continue
                matched = False
                for gi, g in enumerate(gs):
                    if used[gi]:
                        continue
                    if iou(d["bbox"], g["bbox"]) >= 0.5:
                        used[gi] = True
                        matched = True
                        tp += 1
                        break
                if not matched:
                    fp += 1
            fn += sum(1 for u in used if not u)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        rec.append(r)
        prec.append(p)
        f1s.append(f)
    bi = int(np.argmax(f1s))
    return (
        {
            "threshold": round(float(thrs[bi]), 3),
            "precision": round(float(prec[bi]), 4),
            "recall": round(float(rec[bi]), 4),
            "f1": round(float(f1s[bi]), 4),
        },
        {
            "thresholds": [round(float(x), 3) for x in thrs],
            "precision": [round(float(x), 4) for x in prec],
            "recall": [round(float(x), 4) for x in rec],
            "f1": [round(float(x), 4) for x in f1s],
        },
    )


def tiny_ap(cocoGt, cocoDt):
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    ranges = [[0, 16 * 16], [16 * 16, 32 * 32], [32 * 32, 96 * 96], [96 * 96, 1e5]]
    names = ["tiny", "small", "medium", "large"]
    out = {}
    for rg, nm in zip(ranges, names):
        ev.params.areaRng = [[rg[0], rg[1]]]
        ev.params.areaRngLbl = [nm]
        ev.evaluate()
        ev.accumulate()
        out[nm] = round(float(ev.stats[0]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("--out")
    ap.add_argument("--model", "-m", default="?")
    ap.add_argument("--dataset", "-d", default="?")
    ap.add_argument("--params_M", type=float, default=None)
    ap.add_argument("--gflops", type=float, default=None)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    cocoGt = COCO(args.gt)
    raw = json.load(open(args.pred, encoding="utf-8"))
    dets = raw["predictions"] if isinstance(raw, dict) and "predictions" in raw else raw
    cocoDt = cocoGt.loadRes(dets)

    ev = COCOeval(cocoGt, cocoDt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    c7 = coco7(ev)
    bf, prc = best_f1(cocoGt, dets)
    res = {
        "model": args.model,
        "dataset": args.dataset,
        "coco7": c7,
        "best_f1": bf,
        "pr_curve": prc,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if args.params_M is not None:
        res["params_M"] = args.params_M
    if args.gflops is not None:
        res["gflops"] = args.gflops
    if args.tiny:
        res["size_ap"] = tiny_ap(cocoGt, cocoDt)

    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"Model={args.model} Dataset={args.dataset}")
    for k in ["AP50:95", "AP50", "AP75", "AP-s", "AP-m", "AP-l", "AR100"]:
        print(f"  {k}={c7[k]}")
    print(f"  BestF1 thr={bf['threshold']} P={bf['precision']} R={bf['recall']} F1={bf['f1']}")
    print("RESULT: OK")


if __name__ == "__main__":
    main()
