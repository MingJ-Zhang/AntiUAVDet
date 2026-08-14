# DETR R50, single class (drone), 768x1344, 300ep AdamW.
# Thin override on top of the official mmdetection DETR base.
# Requires a clone of mmdetection at detection-baselines/mmdetection/ .

_base_ = ['../../mmdetection/configs/detr/detr_r50_8xb2-500e_coco.py']

img_scale = (1344, 768)

data_root = '${ANTIUAVDET_DATA}/FDD/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False, interpolation='bilinear'),
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
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_foggy_train.json',
        data_prefix=dict(img='train/foggy/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('drone', ))))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_foggy_val.json',
        data_prefix=dict(img='val/foggy/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('drone', ))))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_foggy_test.json',
        data_prefix=dict(img='test/foggy/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('drone', ))))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_foggy_val.json',
    metric='bbox', proposal_nums=[100, 100, 100], classwise=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_foggy_test.json',
    metric='bbox', proposal_nums=[100, 100, 100], classwise=False)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    accumulative_counts=8,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=300, by_epoch=True,
         milestones=[200], gamma=0.1),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True,
                    save_best='coco/bbox_mAP', rule='greater', max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50),
)
randomness = dict(seed=0, deterministic=True, diff_rank_seed=False)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
