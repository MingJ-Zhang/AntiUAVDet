# AntiUAV dataset (COCO format, single class 'UAV').
# Set ANTIUAVDET_DATA to the parent of the antiuav folder.
import os
from pathlib import Path as _P
_DATA = os.environ.get('ANTIUAVDET_DATA', str(_P.home() / 'antiuavdet_data'))
dataset_type = 'CocoDataset'
data_root = str(_P(_DATA) / 'antiuav') + '/'

# mmdet Resize scale is (width, height) -> 1344 x 768
img_scale = (1344, 768)

# Common (non-mosaic) pipeline for Faster R-CNN / RetinaNet / DETR / DINO.
# Photometric jitter + flip; mosaic/mixup are RTMDet-only.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False, interpolation='bilinear'),
    # MMDetection-native HSV/brightness jitter.  Values approximate the
    # YOLO h=0.015, s=0.7, v=0.4 ranges without relying on an unregistered
    # torchvision ColorJitter transform.
    dict(type='PhotoMetricDistortion', brightness_delta=102,
         contrast_range=(0.6, 1.4), saturation_range=(0.3, 1.7),
         hue_delta=3),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=False, interpolation='bilinear'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/img/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('UAV', ), ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/val/img/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('UAV', ), ),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/test/img/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('UAV', ), ),
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    proposal_nums=[100, 100, 100],
    classwise=False,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    proposal_nums=[100, 100, 100],
    classwise=False,
)
