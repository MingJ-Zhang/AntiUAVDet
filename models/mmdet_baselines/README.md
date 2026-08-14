# Detection Baselines (MMDetection)

在 `uav` 项目里搭的一套 **5 模型 × 2 数据集** 检测基线，全部基于 **MMDetection 3.x**，配置尽量与现有 `YOLO-AntiUAV-Benchmark` / `YOLO-FDD-Benchmark` 对齐。

- **模型**：Faster R-CNN、RetinaNet、DETR、DINO、RTMDet
- **数据集**：AntiUAV（单类 `UAV`）、FDD 雾天无人机检测（单类 `drone`）
- **独立目录**：所有内容在 `detection-baselines/`，**不改动其他目录**，配置只读引用服务器现有数据集。

## 1. 为什么用 MMDetection 而不是 Detectron2

五个模型里 RTMDet 是 OpenMMLab 自家的，DETR/DINO 在 MMDetection 有官方实现，Faster R-CNN/RetinaNet 更是标配。Detectron2 根本没有 DETR/DINO/RTMDet。**只有 MMDetection 一套框架能全覆盖**，所以选它。

## 2. 目录结构

```
detection-baselines/
├── configs/
│   ├── _base_/
│   │   ├── antiuav_dataset.py        # AntiUAV COCO 数据/评估 (768x1344)
│   │   ├── fdd_dataset.py            # FDD COCO 数据/评估 (768x1344)
│   │   └── schedule_mmdet_300e.py    # 官方 R-CNN 优化器/调度映射到 300ep
│   ├── faster_rcnn/  (2 配置, 自包含)
│   ├── retinanet/    (2 配置, 自包含)
│   ├── rtmdet/       (2 配置, 自包含, 带 mosaic + 分阶段关闭)
│   ├── detr/         (2 配置, thin override)
│   └── dino/         (2 配置, thin override)
├── scripts/
│   ├── run_antiuav_gpu2.sh           # GPU2 串行跑 5 个 AntiUAV 模型
│   ├── run_fdd_gpu8.sh               # GPU8 串行跑 5 个 FDD 模型
│   ├── run_all.sh                    # nohup 同时启动两个队列
│   └── evaluate.sh                   # 用 best 权重跑 test 评估
├── tools/
│   └── check_dataset.py              # 校验数据路径与 COCO 标注
├── requirements/
│   ├── install_env.sh                # 建 conda env + 装 mmdet + clone mmdetection
│   └── requirements.txt
├── mmdetection/                      # install_env.sh 自动 clone (gitignore)
└── work_dirs/  logs/                 # 训练输出 (gitignore)
```

**两种配置写法**：
- **自包含**（Faster R-CNN / RetinaNet / RTMDet）：`_base_` 只引用本目录的 `antiuav/fdd_dataset.py` + `schedule_mmdet_300e.py`，模型定义完整写在文件里，装好 mmdet 包即可跑。
- **thin override**（DETR / DINO）：transformer 配置极复杂，完整手写易错，所以 `_base_` 直接引用 clone 下来的 `mmdetection/configs/detr|dino/...` 官方配置，只 override `num_classes=1`、数据集路径、768×1344、schedule、预训练权重。**需要 clone mmdetection 仓库**（install_env.sh 自动做）。

## 3. 环境安装（在服务器上跑一次）

```bash
bash ./detection-baselines/requirements/install_env.sh
```

会创建 conda env `mmdet-base`（python 3.10 + torch 2.2.2 + cu121），用 `mim` 装 mmengine/mmcv/mmdet，并把 mmdetection v3.3.0 clone 到 `detection-baselines/mmdetection/` 并 editable 安装。

激活：
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate mmdet-base
```

校验数据（数据在服务器，本地没有）：
```bash
cd ./detection-baselines
python tools/check_dataset.py            # 检查 AntiUAV + FDD 两个
```

## 4. 运行训练

**一键并行（推荐）**：AntiUAV 在 GPU2、FDD 在 GPU8 同时跑，各 5 个模型串行：
```bash
cd ./detection-baselines
bash scripts/run_all.sh
# 跟进日志：
tail -f logs/queue_antiuav_gpu2.log
tail -f logs/queue_fdd_gpu8.log
```

**单数据集**：
```bash
GPU=2 bash scripts/run_antiuav_gpu2.sh   # 可改 GPU=N
GPU=8 bash scripts/run_fdd_gpu8.sh
```

**单实验**（手动指定某个模型+数据集）：
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 \
  python mmdetection/tools/train.py \
  configs/faster_rcnn/faster-rcnn_r50_fpn_1x_antiuav_768x1344.py \
  --work-dir work_dirs/faster_rcnn/antiuav
```

**评估**（训练后，用 val 选出的 best 权重跑 test）：
```bash
bash scripts/evaluate.sh antiuav
bash scripts/evaluate.sh fdd
```

输出位置：
- 权重/日志：`work_dirs/<model>/<dataset>/`（含 `best_coco_bbox_mAP_epoch_XX.pth`、`vis_data/`、`*.json` 指标）
- 训练日志：`logs/<model>_<dataset>_gpuN.log`

## 5. 配置对齐说明（现有 YOLO 实验 ↔ 本套 MMDet）

| 项目 | YOLO benchmark | 本套 MMDet 基线 |
|---|---|---|
| 输入尺寸 | 768(H)×1344(W) | `Resize(scale=(1344,768), keep_ratio=False)` |
| batch | 16 | Faster/RetinaNet/RTMDet=8×accum2=16；DETR/DINO=2×accum8=16 |
| epoch | 300 | 300 |
| 优化器 | AdamW lr=4e-4 wd=1e-4 β1=0.9 | 采用各模型 MMDetection 官方优化器与 batch-scaled LR，见下表 |
| 调度 | 1000 iter warmup，后恒定(lrf=1.0) | 官方调度按比例映射到统一的 300 epoch |
| seed | 0, deterministic | seed=0, deterministic=True |
| 增强 | HSV h.015/s.7/v.4 + scale.5 + translate.1 + fliplr.5 + mosaic.5 + mixup.5 | PhotoMetricDistortion + RandomFlip.5 + 固定 Resize；mosaic/mixup 仅 RTMDet |
| close_mosaic | 最后 12 epoch 关全部强增强 | RTMDet 用 `PipelineSwitchHook` 在 epoch 288 关 mosaic |
| 评估 | COCO AP50:95，conf=0.001, iou=0.7, max_det=100, FP16 | `CocoMetric`，test_cfg score_thr=0.001, nms iou=0.7, max_per_img=100 |
| 选最佳 | val AP50:95 选 best epoch | `CheckpointHook(save_best='coco/bbox_mAP')` |
| 预训练 | COCO 预训练 YOLO | 各模型 `load_from` 对应 COCO 预训练权重（单类头重置） |
| GPU | AntiUAV GPU2 / FDD GPU8 | 同上（`CUDA_VISIBLE_DEVICES`） |

**对齐差异说明**（非完全等价，已尽量贴近）：
1. **mosaic/mixup 只给 RTMDet**：mosaic/mixup 是 YOLO 系增强，对 Faster R-CNN/RetinaNet/DETR/DINO 非常规且可能有害；RTMDet 作为 YOLO 系保留 mosaic + 分阶段关闭。其余模型只用光度扰动和水平翻转，不含 YOLO 的随机 scale/translate。
2. **用梯度累积保持等效 batch 16**：实测 24 GB RTX 3090 上卷积检测器的 batch 16 没有安全余量，因此用 batch 8×accum2；DETR/DINO 用 batch 2×accum8。warmup iter 按 dataloader iteration（每次前向）计，不是 optimizer update。
3. **优化器采用模型官方推荐**：不再强行统一为 YOLO 的 AdamW 4e-4。RTMDet 官方 lr=0.004 对应全局 batch 256，按当前等效 batch 16 线性缩放为 2.5e-4。
4. **单卡 BN**：RTMDet 官方用 SyncBN，单卡改为 BN。
5. **梯度裁剪**：只在 DETR/DINO 保留官方 `max_norm=0.1`；Faster R-CNN、RetinaNet 和 RTMDet 不裁剪。

| 模型 | 优化器 | 300e 调度 |
|---|---|---|
| Faster R-CNN | SGD, lr=0.02, momentum=0.9, wd=1e-4 | warmup 500 iter；epoch 200/275 ×0.1 |
| RetinaNet | SGD, lr=0.01, momentum=0.9, wd=1e-4 | warmup 500 iter；epoch 200/275 ×0.1 |
| DETR | AdamW, lr=1e-4, wd=1e-4；backbone lr×0.1 | epoch 200 ×0.1 |
| DINO | AdamW, lr=1e-4, wd=1e-4；backbone lr×0.1 | epoch 250 ×0.1 |
| RTMDet | AdamW, lr=2.5e-4, wd=0.05；norm/bias wd=0 | warmup 1000 iter；epoch 150–300 cosine 至 5% |

## 6. 各模型配置速查

| 模型 | AntiUAV 配置 | FDD 配置 | 预训练 |
|---|---|---|---|
| Faster R-CNN | `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_antiuav_768x1344.py` | `..._fdd_768x1344.py` | faster-rcnn_r50_fpn_1x_coco |
| RetinaNet | `configs/retinanet/retinanet_r50_fpn_1x_antiuav_768x1344.py` | `..._fdd_768x1344.py` | retinanet_r50_fpn_1x_coco |
| DETR | `configs/detr/detr_r50_8xb2-300e_antiuav_768x1344.py` | `..._fdd_768x1344.py` | detr_r50_8xb2-150e_coco |
| DINO | `configs/dino/dino-4scale_r50_8xb2-300e_antiuav_768x1344.py` | `..._fdd_768x1344.py` | dino-4scale_r50_8xb2-12e_coco |
| RTMDet | `configs/rtmdet/rtmdet_s_8xb16-300e_antiuav_768x1344.py` | `..._fdd_768x1344.py` | rtmdet_s_8xb32-300e_coco |

## 7. 注意事项

- **数据不动**：配置只读引用 `${ANTIUAVDET_DATA}/{antiuav,FDD}`，不改原数据目录。两个数据集都已有现成 COCO json，无需转换。
- **DETR/DINO 依赖 mmdetection 仓库**：thin override 的 `_base_` 指向 `mmdetection/configs/...`，必须先跑 `install_env.sh` clone 仓库。
- **首次跑会下载预训练权重**：`load_from` 的 COCO 权重首次从 openmmlab 下载，需代理（install_env.sh 已设）。
- **本地无数据**：所有训练在服务器跑；本地 `D:\jinshu\uav` 挂载到服务器 `.`，本地改配置 = 服务器立即可见。
- **FP16 推理**：评估若要完全对齐 YOLO 的 FP16，可加 `--cfg-options test_cfg.fp16=True` 或在 test 配置加 `data_preprocessor.half=True`（可选）。
