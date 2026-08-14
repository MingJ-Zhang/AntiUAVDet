from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunnerAdapter:
    """仅描述后端差异；统一协议由顶层配置和中央评估器负责。"""

    model_name: str
    environment: str
    train_script: str
    evaluate_script: str = "evaluate_yolo.py"

    def validate_name(self, requested: str) -> None:
        if requested != self.model_name:
            raise ValueError(f"runner {self.model_name} cannot run {requested}")
