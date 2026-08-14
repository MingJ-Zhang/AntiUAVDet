from __future__ import annotations

from collections.abc import Callable


def disable_online_and_extra_checks() -> None:
    """Keep the benchmark offline and remove version-dependent extra transforms."""
    from ultralytics.data.augment import Albumentations
    from ultralytics.data import utils as data_utils
    from ultralytics.engine import trainer
    from ultralytics.utils import callbacks
    from ultralytics.utils import checks

    Albumentations.__call__ = lambda _self, labels: labels
    trainer.check_amp = lambda _model: True
    # The published FRFDet tree omits ultralytics/hub. Without this guard,
    # integration discovery can import a HUB module from an unrelated checkout
    # on the server's PYTHONPATH. HUB callbacks are not part of local training.
    callbacks.add_integration_callbacks = lambda _instance: None
    # This fork calls ``check_font`` through data.utils while resolving a local
    # dataset. After a server restart with an empty font cache it otherwise
    # blocks on an external URL even when YOLO_OFFLINE is set.
    checks.check_font = lambda *_args, **_kwargs: None
    data_utils.check_font = lambda *_args, **_kwargs: None


def install_fixed_shape(height: int, width: int) -> Callable:
    """Force all train and validation workers to emit fixed H x W tensors."""
    from ultralytics.data import augment
    from ultralytics.data import dataset as dataset_module

    def fixed_transforms(dataset, imgsz: int, hyp, *args, **kwargs):
        del imgsz, args, kwargs
        mosaic = augment.Mosaic(dataset, imgsz=width, p=hyp.mosaic)
        mosaic.border = ((height - 2 * width) // 2, -width // 2)
        affine = augment.RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=augment.LetterBox(new_shape=(height, width)),
        )
        pre_transform = augment.Compose([mosaic, affine])
        flip_idx = dataset.data.get("flip_idx", [])
        return augment.Compose([
            pre_transform,
            augment.MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            augment.Albumentations(p=1.0),
            augment.RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            augment.RandomFlip(direction="vertical", p=hyp.flipud),
            augment.RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ])

    def fixed_build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            transforms = fixed_transforms(self, self.imgsz, hyp)
        else:
            # Prevent the validator's per-batch rectangular shape from overriding
            # the explicit 768x1344 benchmark target in LetterBox.
            self.rect = False
            transforms = augment.Compose([
                augment.LetterBox(new_shape=(height, width), auto=False, scaleup=False)
            ])
        transforms.append(
            augment.Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    augment.v8_transforms = fixed_transforms
    dataset_module.v8_transforms = fixed_transforms
    dataset_module.YOLODataset.build_transforms = fixed_build_transforms
    return fixed_transforms


def register_shape_guard(trainer, height: int, width: int, callback) -> None:
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
