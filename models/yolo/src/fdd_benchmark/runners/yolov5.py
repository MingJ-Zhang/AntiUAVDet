from .base import RunnerAdapter


class YOLOv5Runner(RunnerAdapter):
    def __init__(self) -> None:
        super().__init__("yolov5n", "fdd-yolo", "train_yolov5.py")
