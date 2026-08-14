import os
from pathlib import Path as _P
_DATA = os.environ.get('ANTIUAVDET_DATA', str(_P.home() / 'antiuavdet_data'))
_base_ = './remdet_tiny-300e_768x1344_singleclass.py'

data_root = str(_P(_DATA) / 'antiuav') + '/'
classes = ('UAV', )

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/img/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/val/img/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/test/img/')))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))

