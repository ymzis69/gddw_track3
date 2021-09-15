# -- coding: utf-8
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import csv
import json


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

print("begin to create test labels json!")

seq_root = '../data/track_3/3_testb_images'
label_file = '../data/track_3/3_testb_imageid.csv'
# save_image_root = './visualization'
#
# mkdirs(save_image_root)
save_json_root = '../user_data/tmp_data/3test_b.json'
annotations_info = {'images': [], 'annotations': [], 'categories': []}
# 5 classes
categories_map = {'rg': 1, 'os': 2, 'gs': 3, 'ons': 4, 'gns': 5}
for key in categories_map:
    categoriy_info = {"id": categories_map[key], "name": key}
    annotations_info['categories'].append(categoriy_info)

with open(label_file, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

for i in range(len(rows)-1):
    image_file_name = rows[i+1][1]
    # image_file_name_visual = image_file_name.replace('2_images', 'visualization')
    image_file = cv2.imread(os.path.join(seq_root, image_file_name).replace('3_testb_images/3_testb_images', '3_testb_images'))
    seq_height = image_file.shape[0]
    seq_width = image_file.shape[1]
    seq_height = seq_height
    seq_width = seq_width

    image_info = {'file_name': image_file_name, 'id': i + 1,
                  'height': seq_height, 'width': seq_width}
    annotations_info['images'].append(image_info)


with open(save_json_root, 'w') as f:  # 将信息写入json文件
    json.dump(annotations_info, f, indent=4)

print("create test labels json end!")
