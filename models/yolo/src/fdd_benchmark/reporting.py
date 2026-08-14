from __future__ import annotations

import csv

from .config import BenchmarkConfig
from .io import atomic_json, load_json

REPORT_METRICS = [
    "AP50_95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR100",
]


def _metric_value(row: dict, name: str) -> str:
    item = row.get(name)
    return "-" if item is None else f"{float(item):.2f}"


def _duration(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    seconds = int(value)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _metric_table(rows: list[dict], split: str) -> list[str]:
    labels = ["AP50:95", "AP50", "AP75", "AP-small", "AP-medium", "AP-large", "AR100"]
    lines = [
        f"| 模型 | {' | '.join(labels)} |",
        f"|---|{'|'.join(['---:'] * len(labels))}|",
    ]
    for row in rows:
        values = [_metric_value(row, f"{split}_{name}") for name in REPORT_METRICS]
        lines.append(f"| {row['model']} | {' | '.join(values)} |")
    return lines


def _delta_table(rows: list[dict], baseline: str) -> list[str]:
    reference = next((row for row in rows if row["model"] == baseline), None)
    if reference is None:
        return [f"> {baseline} 尚无完整结果，差值表将在统一评估完成后生成。"]
    lines = [
        "| 模型 | Δ Val AP50:95 | Δ Test AP50:95 | Δ Test AP50 | Δ Test AR100 |",
        "|---|---:|---:|---:|---:|",
    ]
    keys = ("val_AP50_95", "test_AP50_95", "test_AP50", "test_AR100")
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            base = reference.get(key)
            values.append(
                "-"
                if not isinstance(value, int | float) or not isinstance(base, int | float)
                else f"{float(value) - float(base):+.2f}"
            )
        lines.append(f"| {row['model']} | {' | '.join(values)} |")
    return lines


def _reference_audit(cfg: BenchmarkConfig) -> list[str]:
    lines: list[str] = []
    for model in ("deimv2_n", "aode_deim_n"):
        selection = load_json(
            cfg.project_root / "runs" / "reference" / model / "selection.json", {}
        )
        if not selection:
            lines.append(f"- {model}：尚未完成统一验证与 checkpoint 选择。")
            continue
        candidates = selection.get("candidates", [])
        drifts = [
            float(item["max_abs_validation_drift_pp"])
            for item in candidates
            if isinstance(item.get("max_abs_validation_drift_pp"), int | float)
        ]
        drift = max(drifts) if drifts else None
        drift_text = "未记录" if drift is None else f"{drift:.6g} 个百分点"
        lines.append(
            f"- {model}：Val AP50:95 选择 epoch {selection.get('best_epoch', '-')}；"
            f"中央 evaluator 与原日志最大漂移 {drift_text}。"
        )
    return lines


def _read_run(cfg: BenchmarkConfig, model_name: str, reference: bool = False) -> dict | None:
    if reference:
        run = cfg.project_root / "runs" / "reference" / model_name
        val_path = run / "val" / "metrics.json"
        test_path = run / "test" / "metrics.json"
    else:
        run = cfg.project_root / "runs" / model_name
        val_path = run / "evaluation" / "val" / "metrics.json"
        test_path = run / "evaluation" / "test" / "metrics.json"
    selection = load_json(run / "selection.json", {})
    val = load_json(val_path, selection.get("best_val", {}))
    if not val:
        return None
    return {
        "model": model_name,
        "best_epoch": selection.get("best_epoch"),
        "last10_val_AP50_95": selection.get("last10_mean_AP50_95"),
        **{f"val_{key}": value for key, value in val.items()},
        **{f"test_{key}": value for key, value in load_json(test_path, {}).items()},
        **load_json(run / "efficiency.json", {}),
    }


def build_report(cfg: BenchmarkConfig) -> dict:
    rows = [row for model in cfg.models if (row := _read_run(cfg, model))]
    rows.extend(
        row
        for model in ("deimv2_n", "aode_deim_n")
        if (row := _read_run(cfg, model, reference=True))
    )
    output = cfg.project_root / "reports" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "metric_unit": "percentage_points",
        "test_policy": "single evaluation after validation checkpoint selection",
        "rows": rows,
    }
    atomic_json(output / "benchmark_summary.json", payload)
    if rows:
        fields = sorted({key for row in rows for key in row}, key=lambda key: (key != "model", key))
        with (output / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        for baseline in ("deimv2_n", "aode_deim_n"):
            baseline_row = next((row for row in rows if row["model"] == baseline), None)
            if baseline_row is None:
                continue
            deltas: list[dict] = []
            for row in rows:
                delta = {"model": row["model"]}
                for key, value in row.items():
                    reference_value = baseline_row.get(key)
                    if (
                        key != "model"
                        and isinstance(value, int | float)
                        and isinstance(reference_value, int | float)
                    ):
                        delta[key] = float(value) - float(reference_value)
                deltas.append(delta)
            atomic_json(output / f"delta_vs_{baseline}.json", deltas)
            delta_fields = sorted(
                {key for row in deltas for key in row}, key=lambda key: (key != "model", key)
            )
            with (output / f"delta_vs_{baseline}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=delta_fields)
                writer.writeheader()
                writer.writerows(deltas)
            (output / f"delta_vs_{baseline}.md").write_text(
                "\n".join(
                    [
                        f"# 相对 {baseline} 的指标差值（百分点）",
                        "",
                        *_delta_table(rows, baseline),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    lines = [
        "# FDD Foggy YOLO 对比结果",
        "",
        "> 单种子 seed=0；test 仅在 val 选定 checkpoint 后评估一次。所有 AP/AR 均为百分数。",
        "",
        "## Validation",
        "",
        *_metric_table(rows, "val"),
        "",
        "## Test",
        "",
        *_metric_table(rows, "test"),
        "",
        "## 训练与部署效率",
        "",
        "| 模型 | Best epoch | Last-10 Val AP | Params (M) | FLOPs@768×1344 (G) | "
        "训练时长 | 峰值显存 (MiB) | FP16 latency (ms) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {epoch} | {last10} | {params} | {flops} | {duration} | "
            "{memory} | {latency} | {fps} |".format(
                model=row["model"],
                epoch=row.get("best_epoch", "-"),
                last10=_metric_value(row, "last10_val_AP50_95"),
                params=_metric_value(row, "parameters_m"),
                flops=_metric_value(row, "flops_g_768x1344"),
                duration=_duration(row.get("training_wall_seconds")),
                memory=row.get("peak_gpu_memory_mib", "-"),
                latency=_metric_value(row, "latency_e2e_fp16_mean_ms"),
                fps=_metric_value(row, "fps_e2e_fp16"),
            )
        )
    lines.extend(
        [
            "",
            "## 相对参考基线的核心差值",
            "",
            "### DEIMv2-N",
            "",
            *_delta_table(rows, "deimv2_n"),
            "",
            "### AoDE",
            "",
            *_delta_table(rows, "aode_deim_n"),
            "",
        ]
    )
    (output / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")

    report_lines = [
        "# FDD Foggy 六模型公平对比实验报告",
        "",
        "## 实验声明",
        "",
        (
            "本实验仅使用 FDD foggy 单类数据，固定 seed=0。验证集用于选择 checkpoint "
            "和置信度阈值，测试集不参与调参。YOLO 不使用 DEIM 专属 CopyBlend，未做近似替代。"
        ),
        "",
        "## 核心结果",
        "",
        *lines[4:],
        "## 数据与统一协议",
        "",
        (
            "Train/Val/Test 分别为 4599/1438/1150 张 foggy 图像，目标数为 "
            "4544/1424/1117，保留 123/28/45 张真实负样本。四个 YOLO 均使用 "
            "固定 H×W=768×1344 输入、batch 16、300 epochs、AdamW、COCO 预训练、AMP、EMA、seed=0；"
            "最后 12 epochs 关闭强增强。"
        ),
        "",
        "## 选择、阈值与测试隔离",
        "",
        (
            "每轮原图坐标预测由统一 pycocotools evaluator 计算，按 Val AP50:95 唯一选择 "
            "checkpoint。混淆矩阵阈值只在 Val 上按 IoU=0.50 的最大 F1 确定，然后原样用于 "
            "Test；Test 不参与调参且正式预测文件只允许生成一次。"
        ),
        "",
        "## 参考模型一致性审计",
        "",
        *_reference_audit(cfg),
        "",
        "## 可视化与产物",
        "",
        (
            "`figures/` 提供逐模型训练/验证曲线、PR/P/R/F1、原始及归一化混淆矩阵、"
            "六模型固定样本横向预测图，以及 TP/FP/FN/小目标漏检案例；图表同时提供 "
            "PNG/PDF（案例图为 PNG）。"
        ),
        "",
        "## 可复现与审计",
        "",
        (
            "源码 commit、预训练权重 SHA256、数据图像/标注内容 SHA256、冻结依赖、逐轮日志和"
            "状态机 manifest 均随项目保存。当前结论是单种子结果，不包含三种子显著性检验。"
        ),
        "",
    ]
    (output / "REPORT_CN.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload
