import os
from pathlib import Path as _P
_DATA = os.environ.get('ANTIUAVDET_DATA', str(_P.home() / 'antiuavdet_data'))
_base_ = './remdet_tiny-300e_768x1344_singleclass.py'

# DUT-DVE uses the same single-class 768x1344 RemDet-Tiny protocol.
data_root = str(_P(_DATA) / 'DUT-Dve_YOLO_Benchmark_View') + '/'
classes = ('drone', )
max_epochs = 200
close_mosaic_epochs = 12

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/test/')))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2),
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
