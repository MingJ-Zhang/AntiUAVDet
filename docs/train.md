# Training

`antiuavdet train` only **dispatches** the original trainer. Configs keep the published 768×1344 / 300e (DUT-DVE: 200e) recipe.

```bash
export ANTIUAVDET_DATA=/path/to/datasets
antiuavdet train --model <id> --dataset fdd --dry-run
antiuavdet train --model <id> --dataset fdd
```

| Framework | Entry |
|---|---|
| mmdet (except RemDet) | `mim train mmdet <config>` after `pip install -r requirements/mmdet.txt` |
| RemDet | `python models/remdet/tools/train.py <config>` with `PYTHONPATH=models/remdet` |
| RT-DETR | `python models/rtdetr/tools/train.py -c <yml>` |
| D-FINE | `python models/dfine/train.py -c <yml>` |
| DEIM / AoDE | `python models/{deimv2,aode_deim}/train.py -c <yml>` |
| UAV-DETR / FRFDet / DQEF | `python models/<name>/scripts/train_benchmark.py --dataset ...` |
| YOLO v5/8/11/13 | `python -m fdd_benchmark.cli train` from `models/yolo` |

Checkpoints are written under `outputs/` or each model's native `runs/` / `work_dirs/`. Copy the best file to `weights/<id>/<dataset>.pth` (`.pt` for Ultralytics) before `benchmark`.
