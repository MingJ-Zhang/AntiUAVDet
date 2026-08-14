#!/usr/bin/env python3
"""YOLO family (v5n/v8n/11n/13n) test-set inference via fdd_benchmark.cli."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_yolo_predictions(repo: Path, model_id: str):
    candidates = [
        repo / "runs" / model_id / "test_predictions" / "predictions.json",
        repo / "runs" / model_id / "val_predictions" / "test.raw.json",
        repo / "artifacts" / "selected" / f"{model_id}_test.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["fdd", "antiuav", "dut_dve"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    env = os.environ.copy()
    tp = "ultralytics" if args.model != "yolov5n" else "yolov5"
    if args.model == "yolov13n":
        tp = "yolov13"
    env["PYTHONPATH"] = os.pathsep.join([
        str(repo / "src"),
        str(repo / "third_party" / tp),
        env.get("PYTHONPATH", ""),
    ])
    if "cuda" in args.device:
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]

    cli_cfg = repo / "configs" / "benchmark.yaml"
    evaluate_cmd = [
        sys.executable, "-m", "fdd_benchmark.cli",
        "--config", str(cli_cfg), "evaluate",
        "--model", args.model, "--split", "test",
    ]
    print("[infer_yolo]", " ".join(evaluate_cmd))
    if not args.dry_run:
        subprocess.run(evaluate_cmd, cwd=str(repo), env=env, check=True)

    preds = find_yolo_predictions(repo, args.model)
    if preds is None:
        print("[infer_yolo] WARNING: could not auto-locate harness predictions.",
              file=sys.stderr)
        return
    os.makedirs(os.path.dirname(args.out_raw) or ".", exist_ok=True)
    shutil.copy2(preds, args.out_raw)
    print(f"[infer_yolo] copied {preds} -> {args.out_raw}")


if __name__ == "__main__":
    main()
