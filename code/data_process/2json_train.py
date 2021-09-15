# -- coding: utf-8
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import csv
import json
import torch


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '../data/track_3/3_images'
label_file = '../data/track_3/3train_rname.csv'
# save_image_root = './visualization'
#
# mkdirs(save_image_root)
save_json_root = '../user_data/tmp_data/3train.json'
annotations_info = {'images': [], 'annotations': [], 'categories': []}
# 5 classes
categories_map = {'rg': 1, 'os': 2, 'gs': 3, 'ons': 4, 'gns': 5}
for key in categories_map:
    categoriy_info = {"id": categories_map[key], "name": key}
    annotations_info['categories'].append(categoriy_info)

with open(label_file, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

ann_id = 1

print("begin to create train labels json!")

for i in range(len(rows)):
    image_file_name = rows[i][4]
    # image_file_name_visual = image_file_name.replace('2_images', 'visualization')
    image_file = cv2.imread(os.path.join(seq_root, image_file_name).replace('3_images/3_images', '3_images'))
    seq_height = image_file.shape[0]
    seq_width = image_file.shape[1]
    seq_height = seq_height
    seq_width = seq_width

    image_info = {'file_name': image_file_name, 'id': i + 1,
                  'height': seq_height, 'width': seq_width}
    annotations_info['images'].append(image_info)

    image_file_info_item_geometry_for_per_image = []
    image_file_info_item_label_for_per_image = []

    image_file_info = rows[i][5]
    image_file_info = json.loads(image_file_info)
    image_file_info_items = image_file_info["items"]
    for j in range(len(image_file_info_items)):
        image_file_info_item = image_file_info_items[j]
        image_file_info_item_geometry_for_per_image.append(image_file_info_item["meta"]["geometry"])
        image_file_info_item_label_for_per_image.append(image_file_info_item["labels"]["标签"])

    image_id = i + 1

    for k in range(len(image_file_info_item_label_for_per_image)):
        label_person = image_file_info_item_label_for_per_image[k]
        geometry_person = image_file_info_item_geometry_for_per_image[k]

        if label_person == 'ground':
            # 基本信息 #
            xmin, ymin, xmax, ymax = round(geometry_person[0]), round(geometry_person[1]), round(
                geometry_person[2]), round(geometry_person[3])
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            anno_category_id = 5  # gns
            area = w * h
            segmentation = [x, y, w + x, w, w + x, h + y, x, h + y]

            # first, guarder

            for k1 in range(len(image_file_info_item_label_for_per_image)):
                label_person_include = image_file_info_item_label_for_per_image[k1]
                geometry_person_include = image_file_info_item_geometry_for_per_image[k1]

                # compute iof
                person_include_area_w = geometry_person_include[2] - geometry_person_include[0]
                person_include_area_h = geometry_person_include[3] - geometry_person_include[1]
                person_include_area = person_include_area_w * person_include_area_h

                person_iof_area_x1 = max(geometry_person[0], geometry_person_include[0])
                person_iof_area_y1 = max(geometry_person[1], geometry_person_include[1])
                person_iof_area_x2 = min(geometry_person[2], geometry_person_include[2])
                person_iof_area_y2 = min(geometry_person[3], geometry_person_include[3])
                person_iof_area_x = (person_iof_area_x2 - person_iof_area_x1) if (person_iof_area_x2 - person_iof_area_x1) > 0 else 0
                person_iof_area_y = (person_iof_area_y2 - person_iof_area_y1) if (person_iof_area_y2 - person_iof_area_y1) > 0 else 0
                safebelt_overlop = person_iof_area_y * person_iof_area_x

                if geometry_person_include[0] > geometry_person[0] - 12 and geometry_person_include[1] > geometry_person[1] - 12 \
                        and geometry_person_include[2] < geometry_person[2] + 12 and geometry_person_include[3] < geometry_person[3] + 12:
                    if label_person_include == '监护袖章(红only)':  # rg
                        anno_category_id = 1
                        break
                elif safebelt_overlop / person_include_area > 0.24:
                    if label_person_include == 'safebelt':  # gs
                        anno_category_id = 3

            annotation_info = {"segmentation": segmentation, "id": ann_id, "image_id": image_id,
                               "bbox": [x, y, w, h], "category_id": anno_category_id, "area": area, "iscrowd": 0}
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1

        elif label_person == 'offground':
            # 基本信息 #
            xmin, ymin, xmax, ymax = round(geometry_person[0]), round(geometry_person[1]), round(
                geometry_person[2]), round(geometry_person[3])
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            anno_category_id = 4  # ons
            area = w * h
            segmentation = [x, y, w + x, w, w + x, h + y, x, h + y]

            for k1 in range(len(image_file_info_item_label_for_per_image)):
                label_person_include = image_file_info_item_label_for_per_image[k1]
                geometry_person_include = image_file_info_item_geometry_for_per_image[k1]

                # compute iof
                person_include_area_w = geometry_person_include[2] - geometry_person_include[0]
                person_include_area_h = geometry_person_include[3] - geometry_person_include[1]
                person_include_area = person_include_area_w * person_include_area_h

                person_iof_area_x1 = max(geometry_person[0], geometry_person_include[0])
                person_iof_area_y1 = max(geometry_person[1], geometry_person_include[1])
                person_iof_area_x2 = min(geometry_person[2], geometry_person_include[2])
                person_iof_area_y2 = min(geometry_person[3], geometry_person_include[3])
                person_iof_area_x = (person_iof_area_x2 - person_iof_area_x1) if (person_iof_area_x2 - person_iof_area_x1) > 0 else 0
                person_iof_area_y = (person_iof_area_y2 - person_iof_area_y1) if (person_iof_area_y2 - person_iof_area_y1) > 0 else 0
                safebelt_overlop = person_iof_area_y * person_iof_area_x

                if safebelt_overlop / person_include_area > 0.24:
                    if label_person_include == 'safebelt':  # os
                        anno_category_id = 2

            annotation_info = {"segmentation": segmentation, "id": ann_id, "image_id": image_id,
                               "bbox": [x, y, w, h], "category_id": anno_category_id, "area": area, "iscrowd": 0}
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1
    # print(i)

with open(save_json_root, 'w') as f:  # 将信息写入json文件
    json.dump(annotations_info, f, indent=4)

print("create train labels json end!")
