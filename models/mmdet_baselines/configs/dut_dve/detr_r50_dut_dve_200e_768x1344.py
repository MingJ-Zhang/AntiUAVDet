import os
from pathlib import Path as _P
_DATA = os.environ.get('ANTIUAVDET_DATA', str(_P.home() / 'antiuavdet_data'))
_base_ = ['../detr/detr_r50_8xb2-300e_fdd_768x1344.py']

data_root = str(_P(_DATA) / 'DUT-Dve_YOLO_Benchmark_View') + '/'

train_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='images/train/')))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='images/val/')))
test_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='annotations/instances_test2017.json',
    data_prefix=dict(img='images/test/')))

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test2017.json')

train_cfg = dict(max_epochs=200)
