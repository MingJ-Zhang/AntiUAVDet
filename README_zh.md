<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<div align="center">

# AntiUAVDet

### 反无人机小目标检测统一工具箱

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB.svg)](https://www.python.org)
[![Models](https://img.shields.io/badge/models-18-brightgreen.svg)](#模型库)
[![Datasets](https://img.shields.io/badge/datasets-FDD%20%7C%20AntiUAV%20%7C%20DUT--DVE-informational.svg)](#数据集)
[![Input](https://img.shields.io/badge/input-768%C3%971344-orange.svg)](#基准结果)

**18 个检测器 · 5 大框架族 · 3 个无人机数据集 · 同一套 COCO 评测**

</div>

---

AntiUAVDet 把反无人机对比实验里用到的检测器收进同一个命令行。训练和推理走各模型原始实现；评测统一走 wrap → validate → COCO-7。

**不随仓库提供权重。** 请把自行训练或下载的 checkpoint 放到 `weights/`（见 [`weights/README.md`](weights/README.md)）。

## 亮点

- **统一入口：** `antiuavdet list | predict | train | eval | benchmark`
- **18 个检测器**，登记在 [`configs/model_zoo.yaml`](configs/model_zoo.yaml)
- **同一协议：** 输入 `768×1344`、单类、原图像素 xywh、`category_id=0`、每图 top-300、`score ≥ 0.001`
- **框架隔离：** MMDetection、Ultralytics fork、RT-DETR、D-FINE、DEIM 不要混装
- **可审计指标：** 预测只落盘一次，表格全部从磁盘二次加工

## 安装

```bash
git clone https://github.com/MingJ-Zhang/AntiUAVDet.git
cd AntiUAVDet
pip install -e .                 # CLI + COCO 评测（CPU 即可）
```

检测器运行时互不兼容，按框架族各建一个 conda 环境：

```bash
conda create -n antiuav-mmdet python=3.10 -y
conda activate antiuav-mmdet
pip install -r requirements/mmdet.txt

conda create -n antiuav-yolo python=3.10 -y
conda activate antiuav-yolo
pip install -r requirements/yolo.txt

pip install -r requirements/rtdetr.txt   # 或 dfine.txt / deim.txt
```

数据集图片不在本仓库，需自行申请后指定根目录：

```bash
export ANTIUAVDET_DATA=/path/to/datasets
# Windows PowerShell: $env:ANTIUAVDET_DATA = "D:\datasets"
```

目录约定：

```
$ANTIUAVDET_DATA/
  FDD/test/foggy/ ...
  antiuav/test/test/img/ ...
  DUT-Dve_YOLO_Benchmark_View/images/test/ ...
```

## 快速开始

```bash
antiuavdet list

antiuavdet predict --model dfine --source demo.jpg \
    --weights weights/dfine/fdd.pth --device cuda:0

antiuavdet benchmark --model dfine --dataset antiuav \
    --weights weights/dfine/antiuav.pth --device cuda:0

antiuavdet benchmark --model rtmdet --dataset fdd --dry-run
```

## 模型库

Params / FLOPs 在 `768×1344` 下用 mmengine MACs 测量。下表为测试集 `AP50:95`。

| 模型 | 框架 | 参数量 (M) | FLOPs (G) | FDD | AntiUAV | DUT-DVE |
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

完整表见 [`docs/results/`](docs/results/)。

## 数据集

| 数据集 | 测试图 | 类别 | 说明 |
|---|---:|---|---|
| **FDD** | 1150 | drone | 雾天，最难 |
| **AntiUAV** | 2200 | UAV | 个别帧靶牌标注有噪声 |
| **DUT-DVE** | 1200 | drone | 相对简单 |

图片请向原作者申请。本仓库只提供测试集 GT 快照 [`data/gt/`](data/gt/)。详见 [`docs/datasets.md`](docs/datasets.md)。

## 训练

```bash
export ANTIUAVDET_DATA=/path/to/datasets
antiuavdet train --model dfine --dataset fdd --dry-run
antiuavdet train --model dfine --dataset fdd
```

CLI 只负责调度各仓库原始 `train.py`。详见 [`docs/train.md`](docs/train.md)。

## 评测约定

见 [`tools/SCHEMA.md`](tools/SCHEMA.md)：

1. 框必须是**原图像素** xywh
2. `image_id` 取自 GT，禁止用文件序号
3. `category_id` 恒为 0
4. 保存 `score ≥ 0.001`、每图 top-300
5. `images` 与 GT 对齐

`tools/validate_pred.py` 必须输出 `RESULT: PASS`。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/install.md`](docs/install.md) | 分框架环境 |
| [`docs/datasets.md`](docs/datasets.md) | 数据目录 |
| [`docs/train.md`](docs/train.md) | 训练入口 |
| [`docs/infer.md`](docs/infer.md) | 推理评测与 UAV-DETR fp32 协议 |
| [`docs/add_model.md`](docs/add_model.md) | 接入新检测器 |

## 引用

```bibtex
@software{antiuavdet2026,
  title  = {AntiUAVDet: A Unified Toolkit for Anti-UAV Small Object Detection},
  author = {AntiUAVDet contributors},
  year   = {2026},
  url    = {https://github.com/MingJ-Zhang/AntiUAVDet},
  license = {Apache-2.0}
}
```

## 许可证

胶水代码（`antiuavdet/`、`tools/`、文档）为 **Apache-2.0**。`models/` 下第三方检测器保留原许可证（Ultralytics fork 为 AGPL-3.0）。详见 [`NOTICE`](NOTICE)。

## 致谢

[MMDetection](https://github.com/open-mmlab/mmdetection)、
[RT-DETR](https://github.com/lyuwenyu/RT-DETR)、
[D-FINE](https://github.com/Peterande/D-FINE)、
[Ultralytics](https://github.com/ultralytics/ultralytics)、
[RemDet](https://github.com/oaitech/RemDet)、
[DEIM / DEIMv2](https://github.com/Intellindust/DEIM)，
以及 FDD、AntiUAV、DUT-DVE 数据集作者。
