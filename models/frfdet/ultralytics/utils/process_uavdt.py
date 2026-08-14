# 处理UAVDT的脚本, 转成YOLO, COCO, VOC格式
import json
import typing as t
import os
import shutil

import xml.etree.ElementTree as ET

import cv2
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

root = r'C:\dataset\UAVDT'
dataset_dir = r'C:\dataset\UAVDT\UAV-benchmark-M\UAV-benchmark-M'
label_dir = r'C:\dataset\UAVDT\UAV-benchmark-MOTD_v1.0\UAV-benchmark-MOTD_v1.0\GT'
new_label_dir = r'C:\dataset\UAVDT\UAVDT\labels'
new_image_dir = r'C:\dataset\UAVDT\UAVDT\images'
split_dir = r'C:\dataset\UAVDT\M_attr\M_attr'
new_root = r'C:\dataset\UAVDT\UAVDT'

cate_map = {
    1: 'car',
    2: 'truck',
    3: 'bus',
}
true_cate_map = {
    0: 'car',
    1: 'truck',
    2: 'bus',
}

cate_map2 = {
    'car': 0,
    'truck': 1,
    'bus': 2,
}

CATEGORIES_COLOR = {
    'car': [200, 200, 200],
    'truck': [150, 150, 150],
    'bus': [100, 100, 100],
}


def split_and_generate_txt():
    """
    将_gt_whole.txt里面的数据集按照图片名称进行切分，生成图片名对应的txt, 包含所有target的标注信息
    """
    map1 = 0
    maps2 = {}
    new_uavdt_path = os.path.join(root, 'UAVDT')
    uavdt_images_path = os.path.join(new_uavdt_path, 'images')
    uavdt_labels_path = os.path.join(new_uavdt_path, 'labels')
    if not os.path.exists(new_uavdt_path):
        os.makedirs(uavdt_images_path, exist_ok=True)
        os.makedirs(uavdt_labels_path, exist_ok=True)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('whole.txt')]
    label_paths = [os.path.join(label_dir, f) for f in label_files]

    for l_p in label_paths:
        video = os.path.basename(l_p).split('_')[0].strip()
        iframe_num = len(os.listdir(os.path.join(dataset_dir, video)))
        cur_video_label_path = os.path.join(uavdt_labels_path, video)
        os.makedirs(cur_video_label_path, exist_ok=True)
        map1 += len(os.listdir(os.path.join(dataset_dir, video)))
        # 拷贝images

        # shutil.copytree(os.path.join(dataset_dir, video), os.path.join(uavdt_images_path, video))
        # 切分并生成对应图像的txt

        with open(l_p, 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                txt_name = f'img{data[0].zfill(6)}.txt'
                maps2[f'{video}-{txt_name}'] = 1

                # with open(os.path.join(cur_video_label_path, txt_name), 'a+', encoding='utf-8') as new_f:
                #     # object_category, box_left, bbox_top, bbox_width, bbox_height
                #     object_category = data[8].replace('\n', '')
                #     new_f.write(f'{object_category},{data[2]},{data[3]},{data[4]},{data[5]}\n')

        print(f'process {l_p} successfully!')
    # print(maps1, train, test, train + test)
    print(len(maps2))
    print(map1)
    # print(len(train_name) + len(test_name), len(set(train_name + test_name)), len(label_paths))
    # print(sorted([os.path.basename(l).split('_')[0] for l in label_paths]) == sorted(train_name + test_name))
    # a1 = sorted([os.path.basename(l).split('_')[0] for l in label_paths])
    # a2 = sorted(train_name + test_name)
    # w = 1


def split_dataset():
    """
    按照给定txt划分训练集和测试集
    """
    test_dir = os.path.join(split_dir, 'test')
    train_dir = os.path.join(split_dir, 'train')

    test_names = [c.split('_')[0].strip() for c in os.listdir(test_dir)]
    train_names = [c.split('_')[0].strip() for c in os.listdir(train_dir)]

    os.makedirs(os.path.join(new_root, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(new_root, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(new_root, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(new_root, 'test/labels'), exist_ok=True)

    total_train_img_num = 0
    total_train_label_num = 0

    total_test_img_num = 0
    total_test_label_num = 0

    for train_n in train_names:
        shutil.copytree(os.path.join(new_image_dir, train_n),
                        os.path.join(os.path.join(new_root, 'train/images'), train_n))
        shutil.copytree(os.path.join(new_label_dir, train_n),
                        os.path.join(os.path.join(new_root, 'train/labels'), train_n))
        total_train_img_num += len(os.listdir(os.path.join(os.path.join(new_root, 'train/images'), train_n)))
        total_train_label_num += len(os.listdir(os.path.join(os.path.join(new_root, 'train/labels'), train_n)))
        print(f'process {train_n} successfully!')
    for test_n in test_names:
        shutil.copytree(os.path.join(new_image_dir, test_n),
                        os.path.join(os.path.join(new_root, 'test/images'), test_n))
        shutil.copytree(os.path.join(new_label_dir, test_n),
                        os.path.join(os.path.join(new_root, 'test/labels'), test_n))
        total_test_img_num += len(os.listdir(os.path.join(os.path.join(new_root, 'test/images'), test_n)))
        total_test_label_num += len(os.listdir(os.path.join(os.path.join(new_root, 'test/labels'), test_n)))
        print(f'process {test_n} successfully!')
    print('total_train_img_num', total_train_img_num)
    print('total_train_label_num', total_train_label_num)
    print('total_test_img_num', total_test_img_num)
    print('total_test_label_num', total_test_label_num)


def trans_to_coco(root_dir: str, img_dir: str, label_dir: str, mode: str = 'train'):
    """
    转换成COCO格式
    """
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    categories = {}

    # 读取txt解析数据
    video_images_names = os.listdir(img_dir)
    video_label_names = os.listdir(label_dir)

    video_images_names.sort()
    video_label_names.sort()

    assert video_label_names == video_images_names, 'error 111'
    for name in video_images_names:
        cur_image_dir = os.path.join(img_dir, name)
        cur_label_dir = os.path.join(label_dir, name)

        image_names = os.listdir(cur_image_dir)
        # 遍历每个图片
        for img_name in image_names:
            c_img_path = os.path.join(cur_image_dir, img_name)
            c_label_path = os.path.join(cur_label_dir, f'{img_name.split(".")[0]}.txt')
            with Image.open(c_img_path) as img:
                image_width, image_height = img.size
            image_info = {
                'id': len(coco_data['images']) + 1,
                'file_name': f'{name}-{img_name}',
                'width': image_width,
                'height': image_height
            }
            coco_data['images'].append(image_info)

            if os.path.exists(c_label_path):
                annotations = []
                with open(c_label_path, 'r') as f:
                    for line in f.readlines():
                        data = line.split(',')
                        category_id = int(data[0].strip())
                        bbox_left = int(data[1].strip())
                        bbox_top = int(data[2].strip())
                        bbox_width = int(data[3].strip())
                        bbox_height = int(data[4].strip())

                        annotations.append({
                            'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                            'category_id': category_id - 1
                        })
                        categories[category_id - 1] = category_id - 1

                    for anno in annotations:
                        annotation_info = {
                            'id': len(coco_data['annotations']) + 1,
                            'image_id': image_info['id'],
                            'bbox': anno['bbox'],
                            'category_id': anno['category_id'],
                            'iscrowd': 0
                        }
                        coco_data['annotations'].append(annotation_info)
            print(f'process {c_img_path} successfully!')
    for k, v in categories.items():
        coco_data['categories'].append({
            'id': k,
            'name': true_cate_map[int(k)],
            'supercategory': 'unknown'
        })

    with open(os.path.join(root_dir, f'uvadt_{mode}.json'), 'w') as f:
        json.dump(coco_data, f)


def trans_to_voc(anno_dir: str, xml_dir: str):
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
            width = imgs[i]['width']
            height = imgs[i]['height']
            id = imgs[i]['id']

            annotations = [anno for anno in data['annotations'] if anno['image_id'] == id]

            if len(annotations) == 0:
                empty_num += 1
            # xml属性
            xml_content.append("<annotation>")
            xml_content.append("	<folder>UVADT</folder>")
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
            video_name, img_name = file_name.split('-')
            os.makedirs(os.path.join(xml_dir, video_name), exist_ok=True)
            xml_path = os.path.join(os.path.join(xml_dir, video_name), img_name.replace('.jpg', '.xml'))

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
    video_xml_dirs = os.listdir(xml_dir)
    empty_xml = []

    os.makedirs(save_dir, exist_ok=True)
    success_num = 0

    for v_xml_dir in video_xml_dirs:
        cur_v_path = os.path.join(xml_dir, v_xml_dir)
        xml_names = os.listdir(cur_v_path)

        save_txt_dir = os.path.join(save_dir, v_xml_dir)
        os.makedirs(save_txt_dir, exist_ok=True)

        for name in xml_names:
            cur_x_path = os.path.join(cur_v_path, name)
            save_txt_path = os.path.join(save_txt_dir, name.replace('.xml', '.txt'))

            tree = ET.parse(cur_x_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            res = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                cls_id = cate_map2[cls]
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
                empty_xml.append(cur_x_path)
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
    xml_dir_path = r'C:\dataset\UAVDT\UAVDT_THIS\train\xml'
    img_dir_path = r'C:\dataset\UAVDT\UAVDT_THIS\train\images'
    img_out_dir = r'C:\dataset\UAVDT\UAVDT_THIS\vis_gt'
    os.makedirs(img_out_dir, exist_ok=True)
    xml_names = os.listdir(xml_dir_path)
    img_names = os.listdir(img_dir_path)

    for xml_name, img_name in zip(xml_names, img_names):
        xml_path = os.path.join(xml_dir_path, xml_name)
        img_path = os.path.join(img_dir_path, img_name)
        out_path = os.path.join(img_out_dir, img_name)
        vis_validate(img_path, xml_path, out_path)


def rebuild_dataset(data_dir: str, save_dir: str):
    """
    将目录下的文件扁平化拷贝到指定目录
    """
    os.makedirs(save_dir, exist_ok=True)

    cur_lists = os.listdir(data_dir)
    for cur in cur_lists:
        cur_p = os.path.join(data_dir, cur)

        cs_list = os.listdir(cur_p)
        for c in cs_list:
            c_p = os.path.join(cur_p, c)
            shutil.copy(c_p, os.path.join(save_dir, f'{cur}-{c}'))


if __name__ == '__main__':

    pass

    # split_and_generate_txt()
    # split_dataset()
    # trans_to_coco(
    #     r'C:\dataset\UAVDT\UAVDT\train',
    #     r'C:\dataset\UAVDT\UAVDT\train\images', r'C:\dataset\UAVDT\UAVDT\train\labels',
    #     'train'
    # )
    #
    # trans_to_coco(
    #     r'C:\dataset\UAVDT\UAVDT\test',
    #     r'C:\dataset\UAVDT\UAVDT\test\images', r'C:\dataset\UAVDT\UAVDT\test\labels',
    #     'test'
    # )

    # trans_to_voc(r'C:\dataset\UAVDT\UAVDT\train\uvadt_train.json', r'C:\dataset\UAVDT\UAVDT\train\xml')
    # trans_to_voc(r'C:\dataset\UAVDT\UAVDT\test\uvadt_test.json', r'C:\dataset\UAVDT\UAVDT\test\xml')

    # voc_to_yolo(r'C:\dataset\UAVDT\UAVDT\test\xml', r'C:\dataset\UAVDT\UAVDT\test\txt')
    # voc_to_yolo(r'C:\dataset\UAVDT\UAVDT\train\xml', r'C:\dataset\UAVDT\UAVDT\train\txt')

    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\train\images', r'C:\dataset\UAVDT\UAVDT_THIS\train\images')
    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\test\images', r'C:\dataset\UAVDT\UAVDT_THIS\test\images')
    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\train\txt', r'C:\dataset\UAVDT\UAVDT_THIS\train\labels')
    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\test\txt', r'C:\dataset\UAVDT\UAVDT_THIS\test\labels')
    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\train\xml', r'C:\dataset\UAVDT\UAVDT_THIS\train\xml')
    # rebuild_dataset(r'C:\dataset\UAVDT\UAVDT\test\xml', r'C:\dataset\UAVDT\UAVDT_THIS\test\xml')

    # draw_all_labels()
    # r = r'C:\dataset\UAVDT\UAVDT\train\images'
    # all_name = os.listdir(r'C:\dataset\UAVDT\UAV-benchmark-M\UAV-benchmark-M')
    # train_name = [s.split('_')[0].strip() for s in os.listdir(r'C:\dataset\UAVDT\UAVDT\train\labels')]
    # test_name = [s.split('_')[0].strip() for s in os.listdir(r'C:\dataset\UAVDT\UAVDT\test\labels')]
    #
    # train_num = 0
    # test_num = 0
    #
    # for a in all_name:
    #     if a in train_name:
    #         train_num += len(os.listdir(os.path.join(r'C:\dataset\UAVDT\UAVDT\train\labels', a)))
    #     if a in test_name:
    #         test_num += len(os.listdir(os.path.join(r'C:\dataset\UAVDT\UAVDT\test\labels', a)))
    #
    # print('train:', train_num)
    # print('test:', test_num)
    #
    # s = 0
    # for a in os.listdir(r'C:\dataset\UAVDT\UAVDT\labels'):
    #     s += len(os.listdir(os.path.join(r'C:\dataset\UAVDT\UAVDT\labels', a)))
    # print(s)
