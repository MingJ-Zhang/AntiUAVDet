import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

true_cate_map = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
}

true_cate_map2 = {
    'pedestrian': 0,
    'people': 1,
    'bicycle': 2,
    'car': 3,
    'van': 4,
    'truck': 5,
    'tricycle': 6,
    'awning-tricycle': 7,
    'bus': 8,
    'motor': 9,
}


def trans_to_voc(anno_dir: str, xml_dir: str, img_dir: str):
    """
    将COCO 转为 VOC
    """
    num = 0
    empty_num = 0
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)
    with open(anno_dir, 'r') as f:
        data = json.load(f)

        imgs = data['images']
        for i in tqdm(range(len(imgs))):
            xml_content = []
            file_name = imgs[i]['file_name']
            # width = imgs[i]['width']
            # height = imgs[i]['height']
            img_path = os.path.join(img_dir, file_name)
            with Image.open(img_path) as img:
                width, height = img.size
            id = imgs[i]['id']

            annotations = [anno for anno in data['annotations'] if anno['image_id'] == id]

            if len(annotations) == 0:
                empty_num += 1
            # xml属性
            xml_content.append("<annotation>")
            xml_content.append("	<folder>VisDrone-Cluster-Aware</folder>")
            xml_content.append("	<filename>" + file_name + "</filename>")
            xml_content.append("	<size>")
            xml_content.append("		<width>" + str(width) + "</width>")
            xml_content.append("		<height>" + str(height) + "</height>")
            xml_content.append("	</size>")
            xml_content.append("	<segmented>0</segmented>")

            for i, anno in enumerate(annotations):
                bbox = anno['bbox']
                category_id = anno['category_id']
                category_name = true_cate_map[category_id]
                xml_content.append("<object>")
                xml_content.append("<name>" + category_name + "</name>")
                xml_content.append("<pose>Unspecified</pose>")
                xml_content.append("<truncated>0</truncated>")
                xml_content.append("<difficult>0</difficult>")
                xml_content.append("<bndbox>")
                xml_content.append("<xmin>" + str(int(bbox[0])) + "</xmin>")
                xml_content.append("<ymin>" + str(int(bbox[1])) + "</ymin>")
                xml_content.append("<xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
                xml_content.append("<ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
                xml_content.append("</bndbox>")
                xml_content.append("</object>")
            xml_content.append("</annotation>")

            x = xml_content
            xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]
            xml_path = os.path.join(xml_dir, file_name.replace(Path(file_name).suffix, '.xml'))

            with open(xml_path, 'w+', encoding='utf-8') as f:
                f.write('\n'.join(xml_content))
            num += 1

            # print(f'process {xml_path} successfully!')
        print('total num:', num)
        print('empty num:', empty_num)


def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = round(x_center / size[0], 6)
    y = round(y_center / size[1], 6)

    w = round((box[1] - box[0]) / size[0], 6)
    h = round((box[3] - box[2]) / size[1], 6)

    return x, y, w, h


def voc_to_yolo(xml_dir: str, save_dir: str):
    """
    将VOC转为YOLO格式
    """
    xml_names = os.listdir(xml_dir)
    empty_xml = []

    os.makedirs(save_dir, exist_ok=True)
    success_num = 0

    for xml_n in xml_names:
        cur_xml_path = os.path.join(xml_dir, xml_n)
        save_txt_path = os.path.join(save_dir, xml_n.replace('.xml', '.txt'))

        tree = ET.parse(cur_xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls_id = true_cate_map2[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((width, height), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(save_txt_path, 'w+') as f:
                f.write('\n'.join(res))
            success_num += 1
        else:
            empty_xml.append(cur_xml_path)
    print('success', success_num)
    print('empty', len(empty_xml))


def add_suffix(fold: str):
    names = os.listdir(fold)
    i = 0
    for n in names:
        p = os.path.join(fold, n)
        pfix, sfix = p.rsplit('.')
        new_p = f'{pfix}-ca.{sfix}'
        os.rename(p, new_p)
        i += 1
    print('successfully process', i)


if __name__ == '__main__':
    add_suffix(r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone-Cluster-Aware\VisDrone-Cluster-Aware-val\images')
    add_suffix(r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone-Cluster-Aware\VisDrone-Cluster-Aware-val\labels')
    add_suffix(r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone-Cluster-Aware\VisDrone-Cluster-Aware-val\xml')
    # trans_to_voc(
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\annotations\instances_UFP_UAVtrain.json',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\xml_train',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\images\instances_UFP_UAVtrain',
    # )
    #
    # trans_to_voc(
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\annotations\instances_UFP_UAVval.json',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\xml_val',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\images\instances_UFP_UAVval',
    # )

    # voc_to_yolo(
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\xml\train',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\labels\train')
    # voc_to_yolo(
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\xml\val',
    #     r'C:\dataset\cluster-aware-visdrone\VisDrone\VisDrone_Dataset_COCO_Format\VisDrone_Dataset_COCO_Format\labels\val')
