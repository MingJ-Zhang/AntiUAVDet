#!/usr/bin/env python3
"""Aggregate metrics/<ds>/*.json into summary_<ds>.csv/.md."""
import argparse
import csv
import glob
import json
from pathlib import Path

COLS = ["model", "AP50:95", "AP50", "AP75", "AP-s", "AP-m", "AP-l",
        "AR100", "Params(M)", "FLOPs(G)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    metrics_root = root / "metrics"
    out_dir = root / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in ("fdd", "antiuav", "dut_dve"):
        rows = []
        for mp in sorted(glob.glob(str(metrics_root / ds / "*.json"))):
            d = json.load(open(mp, encoding="utf-8"))
            c = d.get("coco7", {})
            rows.append({
                "model": d.get("model", Path(mp).stem),
                "AP50:95": f"{c.get('AP50:95', 0)*100:.2f}",
                "AP50": f"{c.get('AP50', 0)*100:.2f}",
                "AP75": f"{c.get('AP75', 0)*100:.2f}",
                "AP-s": f"{c.get('AP-s', 0)*100:.2f}",
                "AP-m": f"{c.get('AP-m', 0)*100:.2f}",
                "AP-l": f"{c.get('AP-l', 0)*100:.2f}",
                "AR100": f"{c.get('AR100', 0)*100:.2f}",
                "Params(M)": d.get("params_M", ""),
                "FLOPs(G)": d.get("gflops", ""),
            })
        if not rows:
            print(f"[{ds}] no metrics yet")
            continue
        csv_path = out_dir / f"summary_{ds}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        md = out_dir / f"summary_{ds}.md"
        with md.open("w", encoding="utf-8") as f:
            f.write(f"# Benchmark Summary — {ds}\n\n")
            f.write("| " + " | ".join(COLS) + " |\n")
            f.write("|" + "---:|" * len(COLS) + "\n")
            for r in rows:
                f.write("| " + " | ".join(str(r[c]) for c in COLS) + " |\n")
        print(f"[{ds}] wrote {len(rows)} rows -> {md}")


if __name__ == "__main__":
    main()
