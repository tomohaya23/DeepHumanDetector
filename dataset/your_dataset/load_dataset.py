#!/usr/bin/env python
# coding:utf-8

import os
import sys

import numpy as np
from tqdm import tqdm
from utils.data_utils import xyxy2cxcywh


# Example load_data function
def load_data(dataset):
    train_anno_list_file_path = os.path.join(dataset, "train_anno_list.txt")
    if not os.path.isfile(train_anno_list_file_path):
        print(f"'{train_anno_list_file_path}' does not exist.")
        sys.exit(1)

    train_custom_data_label_dict = {}
    with open(train_anno_list_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="[Loading your dataset: train anno list file"):
        file_path, bbox_ltx, bbox_lty, bbox_rbx, bbox_rby, category_id = line.strip().split(",")

        bbox_cx, bbox_cy, bbox_width, bbox_height = xyxy2cxcywh(
            float(bbox_ltx), float(bbox_lty), float(bbox_rbx), float(bbox_rby)
        )

        bbox_info = [bbox_cx, bbox_cy, bbox_width, bbox_height, int(category_id)]
        train_custom_data_label_dict.setdefault(file_path, []).append(bbox_info)

    train_custom_data = list(train_custom_data_label_dict.keys())
    train_custom_label = [
        np.array(label_info, dtype=np.float32) for label_info in train_custom_data_label_dict.values()
    ]

    return train_custom_data, train_custom_label


# To add your own dataset, follow these steps:
#
# 1. Create a new directory for your dataset inside the "data" directory.
#    For example: "data/your_dataset"
#
# 2. Inside your dataset directory, create a subdirectory named "annotations".
#    For example: "data/your_dataset/annotations"
#
# 3. Create a file named "train_anno_list.txt" inside the "annotations" directory.
#    For example: "data/your_dataset/annotations/train_anno_list.txt"
#
# 4. In the "train_anno_list.txt" file, provide the annotations for your dataset in the following format:
#    Each line should contain: file_path, bbox_ltx, bbox_lty, bbox_rbx, bbox_rby, category_id
#    - file_path: The path to the image file.
#    - bbox_ltx, bbox_lty, bbox_rbx, bbox_rby: The coordinates of the bounding box (left-top x, left-top y,
#      right-bottom x, right-bottom y).
#    - category_id: The category ID of the object (e.g., 0 for head, 1 for person).
#
# 5. Create a new Python file named "load_your_dataset.py" inside the "data/your_dataset" directory.
#    For example: "data/your_dataset/load_your_dataset.py"
#
# 6. In the "load_your_dataset.py" file, implement a function named "load_data" that reads the annotations from
#    the "train_anno_list.txt" file and returns the data and labels.
#    The "load_data" function should have the following signature:
#    def load_data(dataset):
#        ...
#        return train_custom_data, train_custom_label
#
#    - The function takes the path to the dataset directory as input.
#    - It returns two values:
#      - train_custom_data: A list of file paths to the training images.
#      - train_custom_label: A list of numpy arrays, where each array contains the bounding box information and
#        category ID for the corresponding image.
#
#    The returned values should have the following data types and formats:
#    - train_custom_data: A list of strings, where each string represents the file path to an image.
#    - train_custom_label: A list of numpy arrays, where each array has the shape (num_boxes, 5) and contains
#      the bounding box information and category ID for each object in the corresponding image.
#      - num_boxes: The number of objects in the image.
#      - Each row of the array represents an object and contains 5 values:
#        [bbox_cx, bbox_cy, bbox_width, bbox_height, category_id].
#        - bbox_cx, bbox_cy: The center coordinates of the bounding box.
#        - bbox_width, bbox_height: The width and height of the bounding box.
#        - category_id: The category ID of the object.
#
# 7. Update the "load_datasets" function in the main script to include your dataset.
#    Add an appropriate condition to check for your dataset and call the corresponding "load_data" function.
#    For example:
#    if dataset == "your_dataset":
#        from data.your_dataset.load_your_dataset import load_data
#        train_data_your, train_label_your = load_data(os.path.join("data", dataset))
#        train_data.extend(train_data_your)
#        train_label.extend(train_label_your)
#
# By following these steps and ensuring that the "load_data" function in your dataset's Python file returns
# the data and labels in the expected format, you can easily add your own dataset to the training pipeline.
