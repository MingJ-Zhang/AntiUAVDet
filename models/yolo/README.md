# YOLO-FDD-Benchmark

FDD 雾天无人机检测公平对比工程。固定在物理 GPU 8 上依次训练 YOLOv5n、
YOLOv8n、YOLO11n、YOLOv13n，并用同一 COCO evaluator 产出 val/test 指标。
正式训练固定 H×W=`768×1344`、batch 16、300 epochs、seed 0。

## 一键执行

本地先执行 `scripts/bootstrap_local.sh`，所有源码、模型与可选 wheel 都保存在本项目内。
服务器端执行：

```bash
ssh symunet "bash ./YOLO-FDD-Benchmark/scripts/setup_remote.sh"
ssh symunet "bash ./YOLO-FDD-Benchmark/scripts/launch_gpu8.sh"
```

GPU 8 已有且经用户明确授权共存的任务时，必须显式开启共享模式；默认仍拒绝抢占：

```bash
ssh symunet "cd ./YOLO-FDD-Benchmark && ALLOW_SHARED_GPU=1 bash scripts/launch_gpu8.sh"
```

状态查看：

```bash
ssh symunet "bash ./YOLO-FDD-Benchmark/scripts/status_remote.sh"
ssh symunet "cat ./YOLO-FDD-Benchmark/runtime/finalization_status.json"
```

`finalize_when_ready.py` 会等待 YOLO 与外部参考训练全部完成，并在 GPU 8 连续空闲后自动执行
参考模型统一评估、六模型可视化、报告、测试和产物校验清单。

完整协议见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)，运维与恢复见
[docs/OPERATIONS.md](docs/OPERATIONS.md)。

## 重要约束

- test 不参与模型选择；仅在 val 锁定最佳 epoch 后运行一次。
- OOM 时禁止自动缩小 batch 或输入尺寸。
- 第三方模型均受强 copyleft 许可证约束，详见许可证说明。
