<div align="center">

# FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion

[![ICME 2026](https://img.shields.io/badge/ICME-2026-blue.svg)](https://2026.ieeeicme.org/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE)

> **FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion**  
> Accepted at **IEEE International Conference on Multimedia and Expo (ICME) 2026**

[📄 Paper](#) &nbsp;|&nbsp; [🏠 Project Page](#) &nbsp;|&nbsp; [📦 Model Zoo](#model-zoo)

</div>

---

## 📌 Overview

Small object detection in UAV imagery remains challenging under adverse conditions including complex weather, low illumination, and sensor noise. These challenges stem from **severe background clutter**, **fine-grained detail degradation**, and **suboptimal semantic–spatial feature fusion**, which jointly hinder robust small-object representation.

**FRFDet** is a lightweight yet effective single-stage detector tailored for UAV-based small object detection. We propose two plug-and-play modules:

- 🔹 **IBS (Inverse Bidirectional Sampling)**: Preserves critical spatial details via channel expansion–compression and bidirectional pattern reconstruction, improving feature alignment across scales.
- 🔹 **SFRCF (Scale-Feature Relationship Cross-Fusion)**: Explicitly models scale-dependent fusion behaviors — inter-group **element-wise multiplication** favors compact models, while inter-group **additive fusion** benefits larger architectures.

Extensive experiments on **VisDrone**, **UAVDT**, **HazyDet**, and **MS COCO** demonstrate that FRFDet achieves state-of-the-art performance among lightweight detectors with **low computational cost**, **compact parameters**, and **fast inference**, making it well suited for resource-constrained UAV platforms.

## 🚀 Getting Started

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU training)

### Installation

**Step 1: Clone this repository**

```bash
git clone https://github.com/YOUR_USERNAME/FRFDet.git
cd FRFDet
```

**Step 2: Install dependencies**

```
To be updated
```

> Current FRFDet is implemented on top of the [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) framework. The core modules `IBS_D`, `IBS_U`, `SFRCFMul`, and `SFRCFAdd` reside in [`nn/FRFDet/FRFDet.py`](nn/FRFDet/FRFDet.py).

---

## 📁 Dataset Preparation

### VisDrone2019-DET

Download from the [official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset) and organize as follows:

```
<your_dataset_root>/
└── VisDrone2019-DET-train/
    └── images/          # 6471 training images
└── VisDrone2019-DET-val/
    └── images/          # 548 validation images
```

Update the `path` field in [`cfg/datasets/VisDrone.yaml`](cfg/datasets/VisDrone.yaml):

```yaml
path: /path/to/your/VisDrone2019-DET
```

### UAVDT

Download from the [UAVDT benchmark](https://sites.google.com/view/grli-uavdt) and organize as:

```
<your_dataset_root>/
├── train/images/        # 24143 training images
├── val/images/          # 16592 validation images
└── test/images/
```

Update the `path` field in [`cfg/datasets/UAVDT.yaml`](cfg/datasets/UAVDT.yaml).

### HazyDet

Download from [HazyDet](https://github.com/GrokCV/HazyDet) and organize as:

```
<your_dataset_root>/
├── train/images/        # 8000 training images
├── val/images/
└── test/images/         # 2000 test images
```

Update the `path` field in [`cfg/datasets/HazyDet.yaml`](cfg/datasets/HazyDet.yaml).

### MS COCO 2017

```bash
bash data/scripts/get_coco.sh
```

---

## 🏋️ Training

### Single GPU

```python
from ultralytics import YOLO

# FRFDet-T/S/M: use SFRCFMul (element-wise multiplication) — compact models
model = YOLO('cfg/models/FRFDet/FRFDet-mul.yaml')
model.train(
    data='cfg/datasets/VisDrone.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
    name='FRFDet-T-VisDrone'
)
```

```python
from ultralytics import YOLO

# FRFDet-L/X: use SFRCFAdd (additive fusion) — larger architectures
model = YOLO('cfg/models/FRFDet/FRFDet-add.yaml')
model.train(
    data='cfg/datasets/VisDrone.yaml',
    epochs=300,
    imgsz=640,
    batch=8,
    device=0,
    name='FRFDet-L-VisDrone'
)
```

### Multi-GPU (DDP)

```bash
python -m torch.distributed.run --nproc_per_node 4 train.py \
    --model cfg/models/FRFDet/FRFDet-mul.yaml \
    --data cfg/datasets/VisDrone.yaml \
    --epochs 300 --imgsz 640 --batch 64
```

### Model Scale Guide

The model scale is controlled via the `scales` field in the YAML config. Choose the fusion variant according to model size:

| Config | Fusion Type | Best For |
|--------|-------------|----------|
| [`FRFDet-mul.yaml`](cfg/models/FRFDet/FRFDet-mul.yaml) | SFRCFMul (×) | T, S, M — compact & efficient deployment |
| [`FRFDet-add.yaml`](cfg/models/FRFDet/FRFDet-add.yaml) | SFRCFAdd (+) | L, X — accuracy-oriented larger models |

Scale suffixes correspond to:
- **T** (Tiny): depth=0.50, width=0.25
- **S** (Small): depth=0.50, width=0.50
- **M** (Medium): depth=0.50, width=1.00
- **L** (Large): depth=1.00, width=1.00
- **X** (XLarge): depth=0.50, width=1.50

---

## 📊 Evaluation

```python
from ultralytics import YOLO

model = YOLO('path/to/best.pt')
metrics = model.val(
    data='cfg/datasets/VisDrone.yaml',
    imgsz=640,
    batch=16,
    device=0,
    split='val'
)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

---

## 🔍 Inference

```python
from ultralytics import YOLO

model = YOLO('path/to/best.pt')
results = model.predict(
    source='path/to/uav_image.jpg',  # image / video / directory
    imgsz=640,
    conf=0.25,
    iou=0.7,
    save=True
)
```

---

## 🔌 Plug-and-Play Integration

IBS and SFRCF are designed as **drop-in plug-and-play modules** for any YOLO-compatible architecture:

```python
from nn.FRFDet import IBS_D, IBS_U, SFRCFMul, SFRCFAdd

# IBS_D: replaces strided convolution in backbone downsampling
ibs_down = IBS_D(
    in_chans=128,
    out_chans=256,
    kernel_size=3,
    stride=2,
    hidden_ratio=2   # channel expansion ratio
)

# IBS_U: replaces bilinear upsampling in neck
ibs_up = IBS_U(
    in_chans=256,
    out_chans=128,
    scale_factor=2,
    hidden_ratio=2
)

# SFRCFMul: cross-scale fusion for compact models (T/S/M)
sfrcf_mul = SFRCFMul(
    in_chans=[256, 512],    # [C_from_small_scale, C_from_large_scale]
    kernel_pair=[3, 5]      # depthwise kernel sizes for two groups
)

# SFRCFAdd: cross-scale fusion for large models (L/X)
sfrcf_add = SFRCFAdd(
    in_chans=[256, 512],
    kernel_pair=[3, 5]
)

# Forward pass
x_down = ibs_down(x)                        # (B, C_in, H, W) → (B, C_out, H/2, W/2)
x_up   = ibs_up(x)                          # (B, C_in, H, W) → (B, C_out, H*2, W*2)
y      = sfrcf_mul([x_small, x_large])      # list of two tensors → fused output
```

---

## 🗂️ Repository Structure

```
FRFDet/
├── cfg/
│   ├── datasets/
│   │   ├── VisDrone.yaml              # VisDrone dataset config (10 classes)
│   │   ├── UAVDT.yaml                 # UAVDT dataset config (3 classes)
│   │   ├── HazyDet.yaml               # HazyDet adverse-weather config (3 classes)
│   │   └── coco.yaml                  # MS COCO 2017 config (80 classes)
│   └── models/
│       └── FRFDet/
│           ├── FRFDet-mul.yaml # FRFDet w/ SFRCFMul (compact models)
│           └── FRFDet-add.yaml # FRFDet w/ SFRCFAdd (large models)
├── nn/
│   └── FRFDet/
│       ├── FRFDet.py                  # ★ Core modules: IBS_D, IBS_U, SFRCFMul, SFRCFAdd
│       └── __init__.py
├── data/                              # Data pipeline (Ultralytics)
├── engine/                            # Training / validation engine (Ultralytics)
├── models/                            # Model wrappers (Ultralytics)
└── README.md
```

---


## 📜 Citation

If you find FRFDet useful in your research, please consider citing:

```bibtex
@article{si2026frfdet,
  title={FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion},
  author={Si, Yunzhong and Xu, Huiying and Zhu, Xinzhong and Liu, Yang and Dong, Yao and Zhang, Wenhao and Li, Hongbo},
  journal={arXiv preprint arXiv:2607.04125},
  year={2026}
}
```

## 🙏 Acknowledgements

This project is built upon the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [MMDetection](https://github.com/open-mmlab/mmdetection) framework (AGPL-3.0). We thank the maintainers of [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [UAVDT](https://sites.google.com/view/grli-uavdt), and [HazyDet](https://github.com/GrokCV/HazyDet) for providing public UAV benchmarks.

---

## 📄 License

This project is released under the [AGPL-3.0 License](LICENSE), consistent with the Ultralytics and MMDetection framework upon which it is built.
