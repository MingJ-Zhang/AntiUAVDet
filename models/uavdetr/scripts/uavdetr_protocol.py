from __future__ import annotations

from collections.abc import Callable


def disable_online_and_extra_checks() -> None:
    """Keep benchmark execution offline and deterministic."""
    from ultralytics.data.augment import Albumentations
    from ultralytics.engine import trainer

    Albumentations.__call__ = lambda _self, labels: labels
    trainer.check_amp = lambda _model: True


def install_fixed_shape(height: int, width: int) -> Callable:
    """Force train and validation workers to emit a fixed H x W tensor."""
    from ultralytics.data import augment
    from ultralytics.data import dataset as dataset_module
    from ultralytics.data.base import BaseDataset
    from ultralytics.models.rtdetr import val as rtdetr_val
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    import torch

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
        flip_idx = dataset.data.get('flip_idx', [])
        return augment.Compose([
            pre_transform,
            augment.MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            augment.Albumentations(p=1.0),
            augment.RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            augment.RandomFlip(direction='vertical', p=hyp.flipud),
            augment.RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx),
        ])

    augment.v8_transforms = fixed_transforms
    dataset_module.v8_transforms = fixed_transforms
    rtdetr_val.v8_transforms = fixed_transforms

    def fixed_load_image(self, index: int, rect_mode: bool = False):
        del rect_mode
        return BaseDataset.load_image(self, index, rect_mode=True)

    rtdetr_val.RTDETRDataset.load_image = fixed_load_image

    def fixed_build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            transforms = fixed_transforms(self, self.imgsz, hyp)
        else:
            transforms = augment.Compose([
                augment.LetterBox(new_shape=(height, width), auto=False, scaleup=False)
            ])
        transforms.append(
            augment.Format(
                bbox_format='xywh',
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    rtdetr_val.RTDETRDataset.build_transforms = fixed_build_transforms

    original_preprocess = DetectionValidator.preprocess

    def rectangular_preprocess(self, batch):
        batch = original_preprocess(self, batch)
        self._benchmark_input_hw = tuple(int(v) for v in batch['img'].shape[2:])
        return batch

    def rectangular_postprocess(self, preds):
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        input_h, input_w = getattr(self, '_benchmark_input_hw', (height, width))
        gain = bboxes.new_tensor((input_w, input_h, input_w, input_h))
        bboxes = bboxes * gain
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for index, bbox in enumerate(bboxes):
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[index].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            outputs[index] = pred[score.argsort(descending=True)][:self.args.max_det]
        return outputs

    def rectangular_update_metrics(self, preds, batch):
        for sample_index, pred in enumerate(preds):
            idx = batch['batch_idx'] == sample_index
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]
            shape = batch['ori_shape'][sample_index]
            correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            self.seen += 1
            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][sample_index].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][sample_index])
            if nl:
                input_h, input_w = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (input_w, input_h, input_w, input_h), device=self.device)
                ops.scale_boxes(batch['img'][sample_index].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][sample_index])
                labelsn = torch.cat((cls, tbox), 1)
                correct = self._process_batch(predn.float(), labelsn)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct, pred[:, 4], pred[:, 5], cls.squeeze(-1)))
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][sample_index])
            if self.args.save_txt:
                from pathlib import Path
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][sample_index]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    rtdetr_val.RTDETRValidator.preprocess = rectangular_preprocess
    rtdetr_val.RTDETRValidator.postprocess = rectangular_postprocess
    rtdetr_val.RTDETRValidator.update_metrics = rectangular_update_metrics
    return fixed_transforms


def register_shape_guard(trainer, height: int, width: int, callback) -> None:
    handle = None

    def guard(_module, inputs) -> None:
        nonlocal handle
        batch = inputs[0]
        image = batch['img'] if isinstance(batch, dict) else batch
        actual = tuple(int(value) for value in image.shape)
        if actual[-2:] != (height, width):
            raise RuntimeError(
                f'protocol input-shape drift: expected (*,{height},{width}), got {actual}')
        callback(actual)
        if handle is not None:
            handle.remove()

    handle = trainer.model.register_forward_pre_hook(guard)
