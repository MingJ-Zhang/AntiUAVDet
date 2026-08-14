#!/usr/bin/env python3
"""AntiUAVDet command-line interface.

    antiuavdet list
    antiuavdet predict --model dfine --source image.jpg --weights path/to.pth
    antiuavdet train   --model dfine --dataset fdd
    antiuavdet eval    --model dfine --dataset fdd --weights path/to.pth
    antiuavdet benchmark --model dfine --dataset fdd --weights path/to.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .paths import ROOT, DATA_ENV, img_dir, gt_path, weights_root, materialize_config
from .registry import (
    load_zoo,
    get_model,
    models,
    active_datasets,
    repo_path,
    config_path,
    weight_path,
)

TOOLS = ROOT / "tools"
PRED_DIR = ROOT / "predictions"
METRICS_DIR = ROOT / "metrics"


def _python() -> str:
    return sys.executable


def _build_env(model: dict) -> dict:
    env = os.environ.copy()
    repo = str(repo_path(model))
    extra = [repo]
    runtime = model.get("pythonpath") or []
    extra.extend(str(ROOT / p) for p in runtime)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([prev] if prev else []))
    env.setdefault("YOLO_OFFLINE", "true")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env.setdefault(DATA_ENV, str(ROOT / "data" / "datasets"))
    return env


def _infer_cmd(model: dict, dataset: str, weights: Path, device: str, score_thr: float, out_raw: Path) -> list[str]:
    fw = model["framework"]
    repo = repo_path(model)
    gt = gt_path(dataset)
    images = img_dir(dataset)
    py = _python()
    mid = model["id"]

    if fw == "mmdet":
        return [
            py, str(TOOLS / "infer_mmdet.py"),
            "--config", str(config_path(model, dataset)),
            "--checkpoint", str(weights),
            "--gt", str(gt), "--img-dir", str(images),
            "--out-raw", str(out_raw),
            "--model", mid, "--dataset", dataset,
            "--device", device, "--score-thr", str(score_thr),
        ]
    if fw == "ultralytics":
        if mid == "uavdetr":
            return [
                py, str(TOOLS / "infer_ultra.py"),
                "--kind", "uavdetr", "--repo", str(repo),
                "--weights", str(weights),
                "--gt", str(gt), "--img-dir", str(images),
                "--out-raw", str(out_raw),
                "--model", mid, "--dataset", dataset,
                "--device", "0" if "cuda" in device else device,
                "--score-thr", str(score_thr),
            ]
        if mid in ("frfdet", "dqef"):
            cmd = [
                py, str(TOOLS / "infer_ultra_yolo.py"),
                "--kind", mid, "--repo", str(repo),
                "--weights", str(weights),
                "--gt", str(gt), "--img-dir", str(images),
                "--out-raw", str(out_raw),
                "--model", mid, "--dataset", dataset,
                "--device", "0" if "cuda" in device else device,
                "--score-thr", str(score_thr), "--max-det", "300",
                "--imgsz", "768,1344",
            ]
            if model.get("infer_opts", {}).get("half"):
                cmd.append("--half")
            return cmd
        return [
            py, str(TOOLS / "infer_yolo.py"),
            "--model", mid, "--dataset", dataset,
            "--repo", str(repo),
            "--gt", str(gt), "--img-dir", str(images),
            "--out-raw", str(out_raw),
            "--device", device, "--score-thr", str(score_thr),
        ]
    if fw == "rt-detr":
        cfg = str(materialize_config(config_path(model, dataset)))
        return [
            py, str(TOOLS / "infer_rtdetr.py"),
            "--repo", str(repo),
            "--config", cfg,
            "--checkpoint", str(weights),
            "--gt", str(gt), "--img-dir", str(images),
            "--out-raw", str(out_raw),
            "--model", mid, "--dataset", dataset,
            "--device", device, "--score-thr", str(score_thr),
        ]
    if fw == "dfine":
        cfg = str(materialize_config(config_path(model, dataset)))
        return [
            py, str(TOOLS / "infer_dfine.py"),
            "--repo", str(repo),
            "--config", cfg,
            "--checkpoint", str(weights),
            "--gt", str(gt), "--img-dir", str(images),
            "--out-raw", str(out_raw),
            "--model", mid, "--dataset", dataset,
            "--device", device, "--size", "768,1344",
            "--score-thr", str(score_thr),
        ]
    if fw == "deim":
        cfg = str(materialize_config(config_path(model, dataset)))
        cmd = [
            py, str(TOOLS / "infer_deim.py"),
            "--repo", str(repo),
            "--config", cfg,
            "--checkpoint", str(weights),
            "--gt", str(gt), "--img-dir", str(images),
            "--out-raw", str(out_raw),
            "--model", mid, "--dataset", dataset,
            "--device", device, "--size", "768,1344",
            "--score-thr", str(score_thr),
        ]
        if model.get("infer_opts", {}).get("normalize"):
            cmd.append("--normalize")
        return cmd
    raise ValueError(f"unsupported framework '{fw}'")


def _run(cmd: list[str], cwd: Path | None, env: dict, dry_run: bool) -> None:
    print("[cmd]", " ".join(str(c) for c in cmd))
    if cwd:
        print("[cwd]", cwd)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def cmd_list(_args) -> None:
    zoo = load_zoo()
    print(f"{'id':22} {'display':26} {'framework':12} datasets")
    for m in models(zoo):
        ds = ",".join(active_datasets(m, zoo))
        print(f"{m['id']:22} {m['display']:26} {m['framework']:12} {ds}")


def cmd_predict(args) -> None:
    """Single-image / folder demo. Requires user-supplied weights."""
    model = get_model(args.model)
    weights = Path(args.weights) if args.weights else weight_path(model, args.dataset or "fdd")
    if not weights.is_file():
        raise SystemExit(
            f"weights not found: {weights}\n"
            f"Place a checkpoint at weights/{model['id']}/<dataset>.pth "
            f"or pass --weights. See weights/README.md."
        )
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    cmd = [
        _python(), str(ROOT / "examples" / "predict_image.py"),
        "--model", model["id"],
        "--weights", str(weights),
        "--source", str(source),
        "--device", args.device,
        "--score-thr", str(args.score_thr),
        "--out", str(args.out or (ROOT / "runs" / "predict")),
    ]
    _run(cmd, cwd=repo_path(model), env=_build_env(model), dry_run=args.dry_run)


def cmd_train(args) -> None:
    model = get_model(args.model)
    if args.dataset in (model.get("skip") or []):
        raise SystemExit(f"{model['id']} has no training config for {args.dataset}")
    env = _build_env(model)
    repo = repo_path(model)
    fw = model["framework"]
    py = _python()
    raw_cfg = config_path(model, args.dataset)
    cfg = str(materialize_config(raw_cfg) if raw_cfg.suffix in {".yml", ".yaml"} else raw_cfg)
    work = ROOT / "outputs" / model["id"] / args.dataset
    if fw == "mmdet":
        native = repo / "tools" / "train.py"
        if native.is_file():
            cmd = [py, str(native), cfg]
        else:
            cmd = ["mim", "train", "mmdet", cfg, "--work-dir", str(work)]
    elif fw == "ultralytics":
        if (repo / "scripts" / "train_benchmark.py").is_file():
            cmd = [py, str(repo / "scripts" / "train_benchmark.py"),
                   "--dataset", args.dataset, "--project-root", str(repo)]
        elif (repo / "scripts" / "train_ultralytics.py").is_file():
            cmd = [py, str(repo / "scripts" / "train_ultralytics.py")]
        else:
            raise SystemExit(f"no train script under {repo}")
    elif fw == "rt-detr":
        cmd = [py, str(repo / "tools" / "train.py"), "-c", cfg]
    elif fw in ("dfine", "deim"):
        cmd = [py, str(repo / "train.py"), "-c", cfg]
    else:
        raise SystemExit(f"train not wired for framework {fw}")
    print(f"[train] {model['id']}/{args.dataset}")
    print(f"        set {DATA_ENV} to your dataset root before launching.")
    _run(cmd, cwd=repo, env=env, dry_run=args.dry_run)


def _stage_infer(model, dataset, weights, device, score_thr, dry_run):
    out_raw = PRED_DIR / dataset / f"{model['id']}.raw.json"
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    cmd = _infer_cmd(model, dataset, weights, device, score_thr, out_raw)
    _run(cmd, cwd=repo_path(model), env=_build_env(model), dry_run=dry_run)
    return out_raw


def _stage_eval(model, dataset, weights, dry_run):
    gt = gt_path(dataset)
    raw = PRED_DIR / dataset / f"{model['id']}.raw.json"
    wrapped = PRED_DIR / dataset / f"{model['id']}.json"
    metrics = METRICS_DIR / dataset / f"{model['id']}.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    epoch = (model.get("best_epoch") or {}).get(dataset, "best")
    wrap = [
        _python(), str(TOOLS / "wrap_pred.py"),
        "--gt", str(gt), "--raw", str(raw), "--out", str(wrapped),
        "--model", model["id"], "--dataset", dataset,
        "--framework", model["framework"],
        "--ckpt", str(weights), "--epoch", str(epoch),
    ]
    validate = [_python(), str(TOOLS / "validate_pred.py"), str(wrapped), str(gt)]
    metrics_cmd = [
        _python(), str(TOOLS / "run_metrics.py"), str(gt), str(wrapped),
        "--out", str(metrics), "--model", model["id"], "--dataset", dataset,
    ]
    if model.get("params_M") is not None:
        metrics_cmd += ["--params_M", str(model["params_M"]), "--gflops", str(model["gflops"])]
    for name, c in (("wrap", wrap), ("validate", validate), ("metrics", metrics_cmd)):
        print(f"[eval:{name}]")
        _run(c, cwd=ROOT, env=os.environ.copy(), dry_run=dry_run)


def cmd_eval(args) -> None:
    model = get_model(args.model)
    weights = Path(args.weights) if args.weights else weight_path(model, args.dataset)
    _stage_eval(model, args.dataset, weights, args.dry_run)


def cmd_benchmark(args) -> None:
    model = get_model(args.model)
    if args.dataset in (model.get("skip") or []):
        print(f"skip {model['id']}/{args.dataset}")
        return
    weights = Path(args.weights) if args.weights else weight_path(model, args.dataset)
    if not args.dry_run and not weights.is_file():
        raise SystemExit(
            f"weights not found: {weights}\n"
            f"Train first, or place a checkpoint under {weights_root() / model['id']}/"
        )
    _stage_infer(model, args.dataset, weights, args.device, args.score_thr, args.dry_run)
    _stage_eval(model, args.dataset, weights, args.dry_run)


def cmd_summarize(args) -> None:
    _run(
        [_python(), str(TOOLS / "summarize.py"), "--root", str(ROOT)],
        cwd=ROOT, env=os.environ.copy(), dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="antiuavdet",
        description="Anti-UAV small object detection toolkit",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list registered models")
    pl.set_defaults(func=cmd_list)

    pp = sub.add_parser("predict", help="run detection on an image or folder")
    pp.add_argument("--model", required=True)
    pp.add_argument("--source", required=True, help="image file or directory")
    pp.add_argument("--weights", default=None)
    pp.add_argument("--dataset", default="fdd", choices=["fdd", "antiuav", "dut_dve"])
    pp.add_argument("--device", default="cuda:0")
    pp.add_argument("--score-thr", type=float, default=0.3)
    pp.add_argument("--out", default=None)
    pp.add_argument("--dry-run", action="store_true")
    pp.set_defaults(func=cmd_predict)

    pt = sub.add_parser("train", help="launch the original training script")
    pt.add_argument("--model", required=True)
    pt.add_argument("--dataset", required=True, choices=["fdd", "antiuav", "dut_dve"])
    pt.add_argument("--dry-run", action="store_true")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="wrap + validate + COCO metrics (needs predictions)")
    pe.add_argument("--model", required=True)
    pe.add_argument("--dataset", required=True, choices=["fdd", "antiuav", "dut_dve"])
    pe.add_argument("--weights", default=None)
    pe.add_argument("--dry-run", action="store_true")
    pe.set_defaults(func=cmd_eval)

    pb = sub.add_parser("benchmark", help="infer a test split then evaluate")
    pb.add_argument("--model", required=True)
    pb.add_argument("--dataset", required=True, choices=["fdd", "antiuav", "dut_dve"])
    pb.add_argument("--weights", default=None)
    pb.add_argument("--device", default="cuda:0")
    pb.add_argument("--score-thr", type=float, default=0.001)
    pb.add_argument("--dry-run", action="store_true")
    pb.set_defaults(func=cmd_benchmark)

    ps = sub.add_parser("summarize", help="rebuild summary tables from metrics/")
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(func=cmd_summarize)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
