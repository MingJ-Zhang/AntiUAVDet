from .base import RunnerAdapter


class YOLOv13Runner(RunnerAdapter):
    def __init__(self) -> None:
        super().__init__("yolov13n", "fdd-yolo13", "train_ultralytics.py")
