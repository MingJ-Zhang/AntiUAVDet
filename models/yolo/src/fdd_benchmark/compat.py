from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path


def prepare_ultralytics_offline_assets(project_root: str | Path) -> None:
    """将已校验字体放入两个上游版本各自的离线配置目录。"""
    root = Path(project_root)
    source = root / "vendor" / "fonts" / "Arial.ttf"
    if not source.is_file():
        raise FileNotFoundError(source)
    config_root = root / ".ultralytics"
    for directory in (config_root, config_root / "Ultralytics"):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)


def enable_pillow_legacy_getsize() -> None:
    """为 YOLOv5 v7.0 补齐 Pillow 10 后移除的只读字体测量接口。"""
    from PIL import ImageFont

    if hasattr(ImageFont.FreeTypeFont, "getsize"):
        return

    def getsize(font, text: str, *args, **kwargs) -> tuple[int, int]:
        left, top, right, bottom = font.getbbox(text, *args, **kwargs)
        return right - left, bottom - top

    ImageFont.FreeTypeFont.getsize = getsize


def disable_ultralytics_extra_augmentations() -> None:
    """禁用未列入公平协议的 Ultralytics Albumentations 默认算子。"""
    from ultralytics.data.augment import Albumentations
    from ultralytics.engine import trainer

    def identity(_transform, labels):
        return labels

    Albumentations.__call__ = identity
    # 上游 AMP 自检会在线加载另一个模型；后端 smoke 已覆盖真实 AMP 前反向。
    trainer.check_amp = lambda _model: True


def install_ultralytics_fixed_shape(
    height: int, width: int, *, yolov13: bool
) -> Callable:
    """让 Ultralytics 训练增强始终输出固定矩形 H×W。

    上游训练 CLI 只接受一个长边并默认构建正方形增强。这里仅替换数据变换
    工厂，模型、损失、优化器和验证器均保持固定版本的原始实现。
    """
    from ultralytics.data import augment
    from ultralytics.data import dataset as dataset_module

    def fixed_transforms(dataset, imgsz: int, hyp, *args, **kwargs):
        del imgsz, args, kwargs
        mosaic = augment.Mosaic(dataset, imgsz=width, p=hyp.mosaic)
        if yolov13:
            # YOLOv13 RandomPerspective 由 mosaic_border 决定输出 H×W。
            mosaic.border = ((height - 2 * width) // 2, -width // 2)
            affine = augment.RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=augment.LetterBox(new_shape=(height, width)),
            )
        else:
            # 新版 RandomPerspective.size 的顺序是 (width, height)。
            affine = augment.RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                size=(width, height),
            )
        pre_transform = augment.Compose([mosaic, affine])
        flip_idx = dataset.data.get("flip_idx", [])
        transforms = [
            pre_transform,
            augment.MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            augment.Albumentations(p=1.0),
            augment.RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            augment.RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            augment.RandomFlip(
                direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx
            ),
        ]
        return augment.Compose(transforms)

    augment.v8_transforms = fixed_transforms
    dataset_module.v8_transforms = fixed_transforms
    return fixed_transforms


def enforce_ultralytics_dataset_shape(dataset, height: int, width: int) -> None:
    """关闭动态 rect_shape，并把 LetterBox 固定到协议尺寸。"""
    from ultralytics.data.augment import Compose, LetterBox

    dataset.rect = False

    def visit(transform) -> None:
        if isinstance(transform, LetterBox):
            transform.new_shape = (height, width)
        if isinstance(transform, Compose):
            for child in transform.transforms:
                visit(child)

    visit(dataset.transforms)


def enforce_ultralytics_validation_shape(trainer, height: int, width: int) -> None:
    """固定训练器内置验证数据集的 H×W。"""
    enforce_ultralytics_dataset_shape(trainer.test_loader.dataset, height, width)


def register_input_shape_guard(trainer, height: int, width: int, callback) -> None:
    """首个真实前向前校验 batch 和 H×W，并记录门禁证据。"""
    handle = None

    def guard(_module, inputs) -> None:
        nonlocal handle
        batch = inputs[0]
        image = batch["img"] if isinstance(batch, dict) else batch
        actual = tuple(int(value) for value in image.shape)
        if actual[-2:] != (height, width):
            raise RuntimeError(
                f"protocol input-shape drift: expected (*,{height},{width}), got {actual}"
            )
        callback(actual)
        if handle is not None:
            handle.remove()

    handle = trainer.model.register_forward_pre_hook(guard)


def disable_yolov13_tensorboard_graph(model) -> None:
    """保留 TensorBoard 标量，跳过 YOLOv13 不稳定且会生成超长告警的 JIT 图。"""
    from ultralytics.utils.callbacks import tensorboard

    # 集成 callback 在 model.train() 内才追加；替换其运行时调用目标最可靠。
    tensorboard._log_tensorboard_graph = lambda _trainer: None
    callbacks = model.callbacks.get("on_train_start", [])
    def is_tensorboard(callback) -> bool:
        module = getattr(callback, "__module__", "")
        name = getattr(callback, "__name__", "")
        wrapped = getattr(callback, "func", None)
        if wrapped is not None:
            module = f"{module} {getattr(wrapped, '__module__', '')}"
            name = f"{name} {getattr(wrapped, '__name__', '')}"
        return "callbacks.tensorboard" in module and "on_train_start" in name

    model.callbacks["on_train_start"] = [
        callback for callback in callbacks if not is_tensorboard(callback)
    ]


def disable_yolov5_extra_augmentations() -> None:
    """禁用未列入公平协议的 YOLOv5 Albumentations 默认算子。"""
    from utils.augmentations import Albumentations

    def identity(_transform, image, labels, probability=1.0):
        return image, labels

    Albumentations.__call__ = identity
