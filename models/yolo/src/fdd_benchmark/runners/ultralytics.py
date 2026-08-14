from .base import RunnerAdapter


class YOLOv8Runner(RunnerAdapter):
    def __init__(self) -> None:
        super().__init__("yolov8n", "fdd-yolo", "train_ultralytics.py")


class YOLO11Runner(RunnerAdapter):
    def __init__(self) -> None:
        super().__init__("yolo11n", "fdd-yolo", "train_ultralytics.py")
