Place user-supplied checkpoints here. Nothing in this folder is committed.

```
weights/<model-id>/<dataset>.pth   # .pt for Ultralytics forks
```

Examples:

```
weights/dfine/fdd.pth
weights/dfine/antiuav.pth
weights/yolov8n/fdd.pt
weights/aode-deim/fdd.pth
```

Override the root with `ANTIUAVDET_WEIGHTS` if you keep weights outside the clone.

```bash
export ANTIUAVDET_WEIGHTS=/mnt/weights
antiuavdet benchmark --model dfine --dataset fdd
```
