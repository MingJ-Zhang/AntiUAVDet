<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<div align="center">

# AntiUAVDet

### Unified Toolkit for Anti-UAV Small Object Detection

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB.svg)](https://www.python.org)
[![Models](https://img.shields.io/badge/models-18-brightgreen.svg)](#model-zoo)
[![Datasets](https://img.shields.io/badge/datasets-FDD%20%7C%20AntiUAV%20%7C%20DUT--DVE-informational.svg)](#datasets)
[![Input](https://img.shields.io/badge/input-768%C3%971344-orange.svg)](#benchmarks)

**18 detectors · 5 framework families · 3 UAV datasets · one COCO metric pipeline**

</div>

---

AntiUAVDet packages the detectors used in a three-dataset anti-UAV benchmark behind a single CLI. Train and infer with the original implementations; evaluate every model with the same wrap → validate → COCO-7 pipeline.

Weights are **not** shipped. Place your own checkpoints under `weights/` (see [`weights/README.md`](weights/README.md)).

## Highlights

- **One command surface.** `antiuavdet list | predict | train | eval | benchmark`
- **18 detectors**, from Faster R-CNN to DEIMv2 / AoDE-DEIM, registered in [`configs/model_zoo.yaml`](configs/model_zoo.yaml)
- **Identical protocol.** Input `768×1344`, single class, original-image xywh, `category_id=0`, top-300, `score ≥ 0.001`
- **Framework isolation.** MMDetection, Ultralytics forks, RT-DETR, D-FINE, and DEIM never share one environment
- **Auditable metrics.** Predictions land on disk once; tables are derived from those files

## Installation

```bash
git clone https://github.com/MingJ-Zhang/AntiUAVDet.git
cd AntiUAVDet
pip install -e .                 # CLI + COCO eval (CPU is enough)
```

Detector runtimes are **not** interchangeable. Create one conda env per family:

```bash
# MMDetection (Faster R-CNN / RetinaNet / DETR / DINO / RTMDet / RemDet)
conda create -n antiuav-mmdet python=3.10 -y
conda activate antiuav-mmdet
pip install -r requirements/mmdet.txt
# RemDet needs its own package on PYTHONPATH (handled by the CLI)

# YOLO-family forks — install torch only; code is vendored under models/
conda create -n antiuav-yolo python=3.10 -y
conda activate antiuav-yolo
pip install -r requirements/yolo.txt

# RT-DETR / D-FINE / DEIM
pip install -r requirements/rtdetr.txt   # or dfine.txt / deim.txt
```

Set the dataset root (images are **not** in this repo):

```bash
export ANTIUAVDET_DATA=/path/to/datasets
# Windows PowerShell: $env:ANTIUAVDET_DATA = "D:\datasets"
```

Expected layout:

```
$ANTIUAVDET_DATA/
  FDD/test/foggy/ ...
  antiuav/test/test/img/ ...
  DUT-Dve_YOLO_Benchmark_View/images/test/ ...
```

## Quick Start

```bash
antiuavdet list

# Single image (you supply the checkpoint)
antiuavdet predict --model dfine --source demo.jpg \
    --weights weights/dfine/fdd.pth --device cuda:0

# Full test-set inference + COCO metrics
antiuavdet benchmark --model dfine --dataset antiuav \
    --weights weights/dfine/antiuav.pth --device cuda:0

# Dry-run: print the exact command without executing
antiuavdet benchmark --model rtmdet --dataset fdd --dry-run
```

## Model Zoo

Params / FLOPs at `768×1344` (mmengine MACs). Test-set `AP50:95`.

| Model | Framework | Params (M) | FLOPs (G) | FDD | AntiUAV | DUT-DVE |
|---|---|---:|---:|---:|---:|---:|
| Faster R-CNN | mmdet | 41.35 | 207.94 | 46.30 | 68.43 | 55.99 |
| RetinaNet | mmdet | 36.33 | 205.56 | 49.83 | 68.91 | 54.96 |
| DETR | mmdet | 41.55 | 98.21 | 44.57 | 63.24 | 47.90 |
| DINO | mmdet | 47.54 | 281.87 | 52.94 | 72.87 | 59.22 |
| RTMDet | mmdet | 8.86 | 37.17 | 53.63 | **75.80** | 60.27 |
| RemDet-Tiny | mmdet | 4.24 | 14.11 | 52.97 | 71.53 | 58.78 |
| RT-DETR | rt-detr | 49.81 | 164.96 | 52.81 | 72.29 | 59.20 |
| UAV-DETR | ultralytics | 11.91 | 81.68 | 52.62 | 70.73 | 54.64 |
| FRFDet | ultralytics | 2.59 | 12.10 | 51.38 | 72.78 | 59.41 |
| DQEF-Net | ultralytics | 2.99 | 12.66 | 53.01 | 72.64 | 59.55 |
| D-FINE | dfine | 3.72 | 8.76 | 53.31 | 72.44 | 55.84 |
| YOLOv5n | ultralytics | 1.77 | 5.26 | 50.84 | 70.84 | 58.11 |
| YOLOv8n | ultralytics | 3.01 | 10.26 | 53.04 | 72.94 | 61.04 |
| YOLO11n | ultralytics | 2.59 | 8.23 | 54.05 | 73.69 | 61.57 |
| YOLOv13n | ultralytics | 2.46 | 10.93 | **54.29** | 73.00 | **61.65** |

Full tables: [`docs/results/`](docs/results/).

## Datasets

| Dataset | Test images | Class | Notes |
|---|---:|---|---|
| **FDD** | 1150 | drone | Foggy scenes, hardest split |
| **AntiUAV** | 2200 | UAV | Occasional GT noise on target boards |
| **DUT-DVE** | 1200 | drone | Relatively easy |

Images must be obtained from the original dataset authors. This repo only ships **test GT snapshots** under [`data/gt/`](data/gt/) so evaluation is reproducible. Details: [`docs/datasets.md`](docs/datasets.md).

## Training

```bash
export ANTIUAVDET_DATA=/path/to/datasets
antiuavdet train --model dfine --dataset fdd --dry-run   # inspect the command
antiuavdet train --model dfine --dataset fdd
```

The CLI launches each model's original `train.py` (or MMDetection / YOLO harness). See [`docs/train.md`](docs/train.md).

## Evaluation contract

See [`tools/SCHEMA.md`](tools/SCHEMA.md):

1. Boxes in **original-image** pixels (xywh)
2. `image_id` from GT, never a file index
3. `category_id ≡ 0`
4. Store `score ≥ 0.001`, top-300 per image
5. `images` aligned with GT

`tools/validate_pred.py` must print `RESULT: PASS`.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/install.md`](docs/install.md) | Environments per framework |
| [`docs/datasets.md`](docs/datasets.md) | How to lay out FDD / AntiUAV / DUT-DVE |
| [`docs/train.md`](docs/train.md) | Training entry points |
| [`docs/infer.md`](docs/infer.md) | Inference, eval, UAV-DETR fp32 protocol |
| [`docs/add_model.md`](docs/add_model.md) | Register a new detector |

## Citation

```bibtex
@software{antiuavdet2026,
  title  = {AntiUAVDet: A Unified Toolkit for Anti-UAV Small Object Detection},
  author = {AntiUAVDet contributors},
  year   = {2026},
  url    = {https://github.com/MingJ-Zhang/AntiUAVDet},
  license = {Apache-2.0}
}
```

## License

Glue code (`antiuavdet/`, `tools/`, docs) is **Apache-2.0**. Third-party detectors under `models/` keep their original licenses (Apache-2.0 or AGPL-3.0 for Ultralytics forks). See [`NOTICE`](NOTICE).

## Acknowledgement

[MMDetection](https://github.com/open-mmlab/mmdetection),
[RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[D-FINE](https://github.com/Peterande/D-FINE),
[Ultralytics](https://github.com/ultralytics/ultralytics),
[RemDet](https://github.com/oaitech/RemDet),
[DEIM / DEIMv2](https://github.com/Intellindust/DEIM),
and the authors of FDD, AntiUAV, and DUT-DVE.
