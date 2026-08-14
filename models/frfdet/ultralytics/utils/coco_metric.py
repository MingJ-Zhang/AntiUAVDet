import json
import os.path
import typing as t
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

FILE_ID_MAP = {}


def map_anno_data(json_file: str) -> t.List:
    with open(json_file, 'r') as f:
        data = json.load(f)

    set_imgid = set()
    for idx, item in enumerate(data['images']):
        set_imgid.add(item['id'])

    set_imgid = sorted(set_imgid)

    map_id_name = {val: idx for idx, val in enumerate(set_imgid)}

    # 原地更新
    for idx, item in enumerate(data['images']):
        data['images'][idx]['id'] = map_id_name[item['id']] + 1
        FILE_ID_MAP[item['file_name'].rsplit('.')[0]] = item['id']

    for idx, item in enumerate(data['annotations']):
        data['annotations'][idx]['image_id'] = map_id_name[data['annotations'][idx]['image_id']]

    return data, map_id_name  # 返回修改后的数据和id映射


def map_pred_data_visdrone(json_file: str) -> t.List:
    with open(json_file, 'r') as f:
        data = json.load(f)

    set_imgid = set()
    for idx, item in enumerate(data):
        set_imgid.add(item['image_id'])

    set_imgid = sorted(set_imgid)

    map_id_name = {val: idx for idx, val in enumerate(set_imgid)}

    new_data = []
    for item in data:
        new_data.append({
            'image_id': map_id_name[item['image_id']],
            'category_id': item['category_id'],
            'bbox': item['bbox'],
            'score': item['score']
        })
    return new_data, map_id_name


def map_pred_data_uavdt(json_file: str) -> t.List:
    with open(json_file, 'r') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_data.append({
            'image_id': FILE_ID_MAP[item['image_id'].replace('-', '/')],
            'category_id': item['category_id'],
            'bbox': item['bbox'],
            'score': item['score']
        })
    return new_data


def main_visdrone(anno_json: str, pred_json: str, new_anno_json: str):
    new_anno_data, a = map_anno_data(anno_json)
    # new_pred_data, b = map_pred_data_visdrone(pred_json)
    with open(pred_json, 'r') as f:
        new_pred_data = json.load(f)

    # 保存新的注释数据到文件
    with open(new_anno_json, 'w') as f:
        json.dump(new_anno_data, f)

    # 创建COCO对象并加载修改后的数据
    anno = COCO(new_anno_json)
    pred = anno.loadRes(new_pred_data)

    # 进行评估
    cocoEval = COCOeval(anno, pred, 'bbox')  # 评估类别为检测框（bbox）
    cocoEval.params.maxDets = [1, 10, 300]
    # cocoEval.params.maxDets = [1, 10, 300]
    print("maxDets:", cocoEval.params.maxDets)
    cocoEval.evaluate()  # 评估
    cocoEval.accumulate()  # 汇总
    cocoEval.summarize()  # 打印总结报告

    # 输出a和b是否相同
    print(a == b)


def main_uavdt(anno_json: str, pred_json: str, new_anno_json: str):
    _, _ = map_anno_data(anno_json)
    new_pred_data = map_pred_data_uavdt(pred_json)

    # 创建COCO对象并加载修改后的数据
    anno = COCO(anno_json)
    pred = anno.loadRes(new_pred_data)

    # 进行评估
    cocoEval = COCOeval(anno, pred, 'bbox')  # 评估类别为检测框（bbox）
    cocoEval.params.maxDets = [1, 10, 300]
    # cocoEval.params.maxDets = [1, 10, 300]
    print("maxDets:", cocoEval.params.maxDets)
    cocoEval.evaluate()  # 评估
    cocoEval.accumulate()  # 汇总
    cocoEval.summarize()  # 打印总结报告


if __name__ == '__main__':
    anno_json = r'C:\dataset\VisDrone2019\annotations\val_visdrone.json'
    pred_json = r'D:\projects\lastestYOLO\ultralytics\runs\detect\result.json'
    new_anno_json = r'C:\dataset\VisDrone2019\annotations\val_visdrone_new.json'
    main_visdrone(anno_json, pred_json, new_anno_json)
