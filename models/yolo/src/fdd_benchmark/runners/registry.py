from .base import RunnerAdapter
from .ultralytics import YOLO11Runner, YOLOv8Runner
from .yolov5 import YOLOv5Runner
from .yolov13 import YOLOv13Runner

RUNNERS: dict[str, RunnerAdapter] = {
    runner.model_name: runner
    for runner in (YOLOv5Runner(), YOLOv8Runner(), YOLO11Runner(), YOLOv13Runner())
}


def get_runner(model_name: str) -> RunnerAdapter:
    try:
        return RUNNERS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model_name}") from error
