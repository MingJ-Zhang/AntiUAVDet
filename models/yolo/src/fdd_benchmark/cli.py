from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import ensure_protocol_lock, load_config
from .data import assert_read_only_source, prepare_dataset_view
from .io import load_json
from .manifest import update_manifest
from .metrics import select_best_epoch
from .reporting import build_report
from .runners import get_runner


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fddbench")
    root.add_argument("--config", default="configs/benchmark.yaml")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("prepare-data")
    for name in ("train", "resume", "select-best", "benchmark"):
        child = commands.add_parser(name)
        child.add_argument("--model", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    state = commands.add_parser("state")
    state.add_argument("--model", required=True)
    state.add_argument(
        "--state",
        choices=("PENDING", "RUNNING", "FAILED", "COMPLETED"),
        required=True,
    )
    state.add_argument("--message", default="")
    state.add_argument("--peak-memory-mib", type=int)
    state.add_argument("--exit-code", type=int)
    state.add_argument("--pid", type=int)
    state.add_argument("--command", dest="run_command")
    state.add_argument("--environment")
    state.add_argument("--gpu", type=int)
    state.add_argument("--data-hash")
    state.add_argument("--log")
    state.add_argument("--resume-point")
    commands.add_parser("status")
    commands.add_parser("report")
    return root


def _run_script(cfg, script: str, model: str, *extra: str) -> None:
    command = [
        sys.executable,
        str(cfg.project_root / "scripts" / script),
        "--config",
        str(cfg.path),
        "--model",
        model,
        *extra,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "preflight":
        assert_read_only_source(cfg)
        report = prepare_dataset_view(cfg)
        ensure_protocol_lock(cfg, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare-data":
        print(json.dumps(prepare_dataset_view(cfg), ensure_ascii=False, indent=2))
    elif args.command in {"train", "resume"}:
        runner = get_runner(args.model)
        extra = ("--resume",) if args.command == "resume" else ()
        _run_script(cfg, runner.train_script, args.model, *extra)
    elif args.command == "select-best":
        result = select_best_epoch(
            cfg.runs_root / args.model,
            cfg.annotation("val"),
            cfg.artifacts_root / "selected" / f"{args.model}.pt",
            expected_epochs=int(cfg.protocol["epochs"]),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        _run_script(cfg, "evaluate_yolo.py", args.model, "--split", args.split)
    elif args.command == "benchmark":
        _run_script(cfg, "benchmark_model.py", args.model)
    elif args.command == "state":
        extra = {"message": args.message}
        if args.peak_memory_mib is not None:
            extra["peak_memory_mib"] = args.peak_memory_mib
        if args.exit_code is not None:
            extra["exit_code"] = args.exit_code
        for name in ("pid", "environment", "gpu", "data_hash", "log", "resume_point"):
            value = getattr(args, name)
            if value is not None:
                extra[name] = value
        if args.run_command is not None:
            extra["command"] = args.run_command
        manifest = update_manifest(
            cfg.runtime_root / "manifest.json",
            args.model,
            args.state,
            **extra,
        )
        print(json.dumps(manifest["models"][args.model], ensure_ascii=False, indent=2))
    elif args.command == "status":
        manifest_path = cfg.runtime_root / "manifest.json"
        print(json.dumps(load_json(manifest_path, {}), ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(json.dumps(build_report(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
