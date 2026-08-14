# Installation

Glue code only:

```bash
pip install -e .
```

That is enough for `antiuavdet list`, wrap / validate / metrics (CPU).

Each detector family needs its **own** environment. Mixing them (especially six Ultralytics forks) will import the wrong `nn` modules.

| Family | File | Python | Notes |
|---|---|---|---|
| mmdet | `requirements/mmdet.txt` | 3.10 | torch 2.2.2 + mmcv 2.2 + mmdet 3.3. RemDet uses `PYTHONPATH=models/remdet` |
| yolo forks | `requirements/yolo.txt` | 3.10 | Do **not** `pip install ultralytics`. Code lives under `models/` |
| rt-detr | `requirements/rtdetr.txt` | 3.10 | Official `rtdetr_pytorch` |
| dfine | `requirements/dfine.txt` | 3.10 | |
| deim | `requirements/deim.txt` | 3.10 | DEIMv2 + AoDE-DEIM; DINOv3 weights download on first train |

```bash
export ANTIUAVDET_DATA=/path/to/datasets
export ANTIUAVDET_WEIGHTS=/path/to/weights   # optional, default ./weights
```

Self-check (no GPU):

```bash
python tests/test_layout.py
```
