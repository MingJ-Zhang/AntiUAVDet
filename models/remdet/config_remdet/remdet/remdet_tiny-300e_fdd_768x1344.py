import os
from pathlib import Path as _P
_DATA = os.environ.get('ANTIUAVDET_DATA', str(_P.home() / 'antiuavdet_data'))
_base_ = './remdet_tiny-300e_768x1344_singleclass.py'

data_root = str(_P(_DATA) / 'FDD') + '/'
classes = ('drone', )

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_foggy_train.json',
        data_prefix=dict(img='train/foggy/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_foggy_val.json',
        data_prefix=dict(img='val/foggy/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_foggy_test.json',
        data_prefix=dict(img='test/foggy/')))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_foggy_val.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_foggy_test.json',
    metric='bbox',
    proposal_nums=(1, 10, 100))

