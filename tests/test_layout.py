#!/usr/bin/env python3
"""No-GPU layout check. Exit 1 on hard errors."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antiuavdet.registry import load_zoo, models, active_datasets, repo_path, config_path

STRAY = re.compile(r"/mnt/ws/uav|D:[/\\]jinshu|/home/user/jinshu")
ADAPTER = {
    "mmdet": "infer_mmdet.py",
    "rt-detr": "infer_rtdetr.py",
    "dfine": "infer_dfine.py",
    "deim": "infer_deim.py",
}


def infer_script(m: dict) -> str:
    fw = m["framework"]
    if fw == "ultralytics":
        if m["id"] == "uavdetr":
            return "infer_ultra.py"
        if m["id"] in ("frfdet", "dqef"):
            return "infer_ultra_yolo.py"
        return "infer_yolo.py"
    return ADAPTER[fw]


def main() -> int:
    errors, warns = [], []
    zoo = load_zoo()
    ms = models(zoo)
    if len(ms) < 17:
        errors.append(f"expected >=17 models, got {len(ms)}")

    for m in ms:
        mid = m["id"]
        repo = repo_path(m)
        if not repo.is_dir():
            errors.append(f"[{mid}] repo missing: {repo}")
        script = ROOT / "tools" / infer_script(m)
        if not script.is_file():
            errors.append(f"[{mid}] adapter missing: {script.name}")
        for ds in active_datasets(m, zoo):
            try:
                cfg = config_path(m, ds)
            except KeyError as e:
                errors.append(str(e))
                continue
            if not cfg.is_file():
                errors.append(f"[{mid}/{ds}] config missing: {cfg}")

    for folder in (ROOT / "antiuavdet", ROOT / "tools", ROOT / "configs"):
        for py in folder.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                if STRAY.search(line) and "sanitize" not in py.name and "_extract" not in py.name:
                    warns.append(f"absolute path {py.relative_to(ROOT)}:{i}")

    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in ("*.pth", "*.pt", "weights/"):
        if token not in gi:
            errors.append(f".gitignore missing {token}")

    for name in ("fdd_test.json", "antiuav_test.json", "dut_dve_test.json"):
        if not (ROOT / "data" / "gt" / name).is_file():
            errors.append(f"missing GT {name}")

    print(f"models={len(ms)} errors={len(errors)} warnings={len(warns)}")
    for e in errors:
        print("ERROR", e)
    for w in warns[:30]:
        print("WARN", w)
    if errors:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
