# Inference and evaluation

```bash
antiuavdet predict --model dfine --source image.jpg --weights weights/dfine/fdd.pth
antiuavdet benchmark --model dfine --dataset antiuav --weights weights/dfine/antiuav.pth
```

Pipeline:

```
infer_*.py  →  predictions/<ds>/<model>.raw.json
wrap_pred.py  →  <model>.json
validate_pred.py  →  RESULT: PASS
run_metrics.py  →  metrics/<ds>/<model>.json
```

Contract: [`tools/SCHEMA.md`](../tools/SCHEMA.md).

## UAV-DETR

Must run **fp32** with `install_fixed_shape(768, 1344)`. `fp16` + WTConv + scaleFill yields giant boxes and AP ≈ 0. The shipped `tools/infer_ultra.py` already enforces this.

## DEIMv2 normalize

| Model | `--normalize` |
|---|---|
| DEIMv2-DINOv3-M | **on** (ImageNet mean/std) |
| DEIMv2-HGNetv2-N | off |
| AoDE-DEIM | off |

The CLI reads `infer_opts.normalize` from `configs/model_zoo.yaml`.

## YOLO family

`tools/infer_yolo.py` calls `python -m fdd_benchmark.cli evaluate`. `models/yolo/src/fdd_benchmark` is required; `third_party/` alone is not enough.
