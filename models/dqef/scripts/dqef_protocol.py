from __future__ import annotations

from collections.abc import Callable


def disable_online_and_extra_checks() -> None:
    """Keep the benchmark offline and remove non-paper, version-dependent transforms."""
    from ultralytics.data.augment import Albumentations
    from ultralytics.engine import trainer

    Albumentations.__call__ = lambda _self, labels: labels
    trainer.check_amp = lambda _model: True


def install_fixed_shape(height: int, width: int) -> Callable:
    """Force every train/validation worker to emit the fixed H x W benchmark tensor."""
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
            # The validator creates a rectangular dataset and injects a per-batch
            # ``rect_shape`` into each sample.  That value otherwise overrides the
            # explicit LetterBox target below (e.g. 800x1376 instead of 768x1344).
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
