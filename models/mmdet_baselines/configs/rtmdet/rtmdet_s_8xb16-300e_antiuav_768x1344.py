# RTMDet-s, single class (UAV), 768x1344, 300ep AdamW, effective batch 16.
# Thin override on top of the official mmdetection RTMDet-s base.
# Requires a clone of mmdetection at detection-baselines/mmdetection/ .
_base_ = ['../../mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py']

img_scale = (1344, 768)  # (W, H)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=img_scale, ratio_range=(1.0, 1.0),
         max_cached_images=20, pad_val=(114, 114, 114)),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
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

# single class + BN (official base uses SyncBN, not runnable on single GPU)
model = dict(
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head=dict(num_classes=1, norm_cfg=dict(type='BN', requires_grad=True)),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=100),
)

data_root = '${ANTIUAVDET_DATA}/antiuav/'
train_dataloader = dict(
    batch_size=8, num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/img/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('UAV', ))))
val_dataloader = dict(
    batch_size=1, num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/val/img/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('UAV', ))))
test_dataloader = dict(
    batch_size=1, num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/test/img/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('UAV', ))))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox', proposal_nums=[100, 100, 100], classwise=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox', proposal_nums=[100, 100, 100], classwise=False)

# Official RTMDet 300e optimizer/scheduler. The official lr=0.004 is for
# global batch 256; linearly scaled to effective batch 16 -> 0.00025.
base_lr = 2.5e-4
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    accumulative_counts=2,
    clip_grad=None,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True,
         convert_to_iter_based=True),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# EMA (aligned with YOLO) + close mosaic last 12 epochs (switch at 288)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook', switch_epoch=288,
         switch_pipeline=train_pipeline_stage2),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True,
                    save_best='coco/bbox_mAP', rule='greater', max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50),
)
randomness = dict(seed=0, deterministic=True, diff_rank_seed=False)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
