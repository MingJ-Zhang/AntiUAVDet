import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

true_cate_map = {
    0: 'car',
    1: 'truck',
    2: 'bus',
}

true_cate_map2 = {
    'car': 0,
    'truck': 1,
    'bus': 2,
}

CATEGORIES_COLOR = {
    'car': [200, 200, 200],
    'truck': [150, 150, 150],
    'bus': [100, 100, 100],
}


def trans_to_voc(anno_dir: str, xml_dir: str, img_dir: str = ''):
    """
    将COCO 转为 VOC
    """
    num = 0
    empty_num = 0
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(anno_dir, 'r') as f:
        data = json.load(f)

        imgs = data['images']
        for i in tqdm(range(len(imgs))):
            xml_content = []
            file_name = imgs[i]['file_name']
            s_file_name = file_name.replace('/', '-')
            img_path = os.path.join(r'C:\dataset\cluster-aware-uvadt\UAVDT\images\UAV-benchmark-M', file_name)

            shutil.copy(img_path,
                        os.path.join(img_dir, s_file_name))
            with Image.open(img_path) as img:
                width, height = img.size

            # width = imgs[i]['width']
            # height = imgs[i]['height']
            #
            # assert width == t_width and height == t_height, 'width or height is not same with truth'

            id = imgs[i]['id']

            annotations = [anno for anno in data['annotations'] if anno['image_id'] == id]

            if len(annotations) == 0:
                empty_num += 1
            # xml属性
            xml_content.append("<annotation>")
            xml_content.append("	<folder>UAVDT-Cluster-Aware</folder>")
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
            xml_path = os.path.join(xml_dir, s_file_name.replace(Path(s_file_name).suffix, '.xml'))

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


def vis_validate(image_path: str, annotation_xml: str, save_path: str):
    """
    Visually verify if the rotated labels are aligned with the rotated image.
    """
    # 解析XML文件
    tree = ET.parse(annotation_xml)
    root = tree.getroot()

    # 遍历每个图像元素
    boxes = []
    cls_names = []
    for image_elem in root.iter('object'):
        # image_file = image_elem.get('file')
        xmlbox = image_elem.find('bndbox')
        cls_name = image_elem.find('name').text
        box = {
            'xmin': xmlbox.find('xmin').text,
            'ymin': xmlbox.find('ymin').text,
            'xmax': xmlbox.find('xmax').text,
            'ymax': xmlbox.find('ymax').text
        }
        boxes.append(box)
        cls_names.append(cls_name)

    # 读取图像
    image = cv2.imread(image_path)
    # 如果图像为空，返回
    if image is None:
        print("Error loading image")
        return

    # 在图像上绘制每个边界框
    for box, cls_name in zip(boxes, cls_names):
        left = int(float(box['xmin']))
        top = int(float(box['ymin']))
        right = int(float(box['xmax']))
        bottom = int(float(box['ymax']))

        # 绘制矩形框
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        name = cls_name
        # if cls_name not in CATEGORIES_COLOR:
        #     name = cate_map[cls_name]
        # else:
        #     name = cls_name
        cv2.putText(image, cls_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, CATEGORIES_COLOR[name], 2)

    # 显示图像
    # image = cv2.resize(image, (1280, 1280))
    # cv2.imshow("Image with Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 或者保存图像
    cv2.imwrite(save_path.replace('.bmp', '.jpg'), image)


def draw_all_labels():
    xml_dir_path = r'C:\dataset\cluster-aware-uvadt\UAVDT\train\xml'
    img_dir_path = r'C:\dataset\cluster-aware-uvadt\UAVDT\train\images'
    img_out_dir = r'C:\dataset\cluster-aware-uvadt\UAVDT\vis_gt'
    os.makedirs(img_out_dir, exist_ok=True)
    xml_names = os.listdir(xml_dir_path)
    img_names = os.listdir(img_dir_path)

    for xml_name, img_name in zip(xml_names, img_names):
        xml_path = os.path.join(xml_dir_path, xml_name)
        img_path = os.path.join(img_dir_path, img_name)
        out_path = os.path.join(img_out_dir, img_name)
        vis_validate(img_path, xml_path, out_path)


if __name__ == '__main__':
    # trans_to_voc(r'C:\dataset\cluster-aware-uvadt\UAVDT\annotations\UAV-benchmark-M-Train.json',
    #              r'C:\dataset\cluster-aware-uvadt\UAVDT\xml\train',
    #              r'C:\dataset\cluster-aware-uvadt\UAVDT\temp_image\train',
    #              )
    #
    # trans_to_voc(r'C:\dataset\cluster-aware-uvadt\UAVDT\annotations\UAV-benchmark-M-Val.json',
    #              r'C:\dataset\cluster-aware-uvadt\UAVDT\xml\val',
    #              r'C:\dataset\cluster-aware-uvadt\UAVDT\temp_image\val',
    #              )

    # voc_to_yolo(r'C:\dataset\cluster-aware-uvadt\UAVDT\xml\train',
    #             r'C:\dataset\cluster-aware-uvadt\UAVDT\labels\train'
    #             )
    #
    # voc_to_yolo(r'C:\dataset\cluster-aware-uvadt\UAVDT\xml\val',
    #             r'C:\dataset\cluster-aware-uvadt\UAVDT\labels\val'
    #             )
    draw_all_labels()
