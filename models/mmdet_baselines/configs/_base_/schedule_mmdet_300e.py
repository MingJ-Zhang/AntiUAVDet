# MMDetection R-CNN 1x schedule scaled from 12 to 300 epochs.
# This is the Faster R-CNN default; RetinaNet overrides lr to 0.01.
# Physical batch 8 x gradient accumulation 2 = effective batch 16.

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    # Physical batch 8 x2 = 16 effective. DETR/DINO override this to x8.
    accumulative_counts=2,
    clip_grad=None,
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    # Official 1x milestones [8, 11] scaled by 300 / 12.
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        by_epoch=True,
        milestones=[200, 275],
        gamma=0.1,
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# mimic YOLO: deterministic + seed 0
default_scope = 'mmdet'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,           # save every epoch (save_period=1)
        by_epoch=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
    ),
    sync_buffers=dict(type='SyncBuffersHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
log_level = 'INFO'
load_from = None
resume = False

# seed
randomness = dict(seed=0, deterministic=True, diff_rank_seed=False)
