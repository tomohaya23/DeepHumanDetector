#!/usr/bin/env python
# coding:utf-8

from random import sample, shuffle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import cxcywh2xyxy, imread, xyxy2cxcywh


class HumanDetectionDataset(Dataset):
    def __init__(
        self,
        image_path_list,
        label_list,
        width,
        height,
        mosaic,
        mosaic_prob,
        mixup,
        mixup_prob,
        training,
    ):
        """
        Initialize the HumanDetectionDataset.

        Args:
            image_path_list (list): List of image paths.
            label_list (list): List of labels.
            width (int): Width of the input image.
            height (int): Height of the input image.
            mosaic (bool): Whether to apply mosaic augmentation.
            mosaic_prob (float): Probability of applying mosaic augmentation.
            mixup (bool): Whether to apply mixup augmentation.
            mixup_prob (float): Probability of applying mixup augmentation.
            training (bool): Whether the dataset is used for training or not.
        """
        super(HumanDetectionDataset, self).__init__()
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.width = width
        self.height = height
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.training = training

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.image_path_list)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the image data and label.
        """

        image_path = self.image_path_list[idx]
        image_data = imread(image_path)
        label = self.label_list[idx]
        image_data, label = self.preprocess_data(image_data, label, idx)
        image_data = torch.from_numpy(image_data).to(dtype=torch.float32)
        label = torch.from_numpy(label).to(dtype=torch.float32)
        return image_data, label

    def preprocess_data(self, image_data, label, idx):
        """
        Preprocess the image data and label.

        Args:
            image_data (np.ndarray): Image data.
            label (np.ndarray): Label data.
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the preprocessed image data and label.
        """

        # Correct protruding coordinates
        label = self.correct_protruding_coordinates(image_data.size, label)

        # Data augmentation
        if self.training:
            if self.mosaic and self.rand() < self.mosaic_prob:
                image_data_list, label_list = self.pick_up_sample_data(sample_num=3, idx=idx)
                image_data_list.append(image_data)
                label_list.append(label)
                idx_list = list(range(0, len(image_data_list)))
                shuffle(idx_list)
                image_data_list = [image_data_list[idx] for idx in idx_list]
                label_list = [label_list[idx] for idx in idx_list]
                dst_image_data, dst_label = self.get_mosaic_data(image_data_list, label_list)
                image_data_list = None
                label_list = None

                if self.mixup and self.rand() < self.mixup_prob:
                    image_data_list, label_list = self.pick_up_sample_data(sample_num=1, idx=idx)
                    dst_image_data2, dst_label2 = self.scale_image_with_aspect_ratio(image_data_list[0], label_list[0])
                    dst_image_data, dst_label = self.get_mixup_data(
                        dst_image_data, dst_label, dst_image_data2, dst_label2
                    )
                    image_data_list = None
                    label_list = None
            else:
                dst_image_data, dst_label = self.scale_image_with_aspect_ratio(image_data, label)
        else:
            dst_image_data, dst_label = self.scale_image_with_aspect_ratio(image_data, label)

        dst_image_data = dst_image_data / 255
        dst_image_data = np.transpose(dst_image_data, (2, 0, 1))

        return dst_image_data, dst_label

    def get_mixup_data(self, image_1, box_1, image_2, box_2):
        """
        Apply mixup augmentation to the image data and labels.

        Args:
            image_1 (np.ndarray): First image data.
            box_1 (np.ndarray): First bounding box data.
            image_2 (np.ndarray): Second image data.
            box_2 (np.ndarray): Second bounding box data.

        Returns:
            tuple: Tuple containing the mixup image data and bounding boxes.
        """

        mixup_image = image_1 * 0.5 + image_2 * 0.5
        if len(box_1) == 0:
            mixup_boxes = box_2
        elif len(box_2) == 0:
            mixup_boxes = box_1
        else:
            mixup_boxes = np.concatenate([box_1, box_2], axis=0)

        return mixup_image, mixup_boxes

    def merge_bboxes(self, bboxes, cutx, cuty):
        """
        Merge bounding boxes from different images after mosaic augmentation.

        Args:
            bboxes (list): List of bounding box arrays.
            cutx (int): X-coordinate of the mosaic split position.
            cuty (int): Y-coordinate of the mosaic split position.

        Returns:
            list: List of merged bounding boxes.
        """

        merge_bbox = []

        for i in range(len(bboxes)):
            for bbox in bboxes[i]:
                tmp_box = []
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

                # Adjust the bounding box based on the mosaic image split position
                if i == 0:  # Top-left image
                    if ymin > cuty or xmin > cutx:  # Box is protruding to the bottom-left or top-right
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymax = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmax = cutx

                if i == 1:  # Bottom-left image
                    if ymax < cuty or xmin > cutx:  # Box is protruding to the top-left or bottom-right
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymin = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmax = cutx

                if i == 2:  # Bottom-right image
                    if ymax < cuty or xmax < cutx:  # Box is protruding to the top-right or bottom-left
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymin = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmin = cutx

                if i == 3:  # Top-right image
                    if ymin > cuty or xmax < cutx:  # Box is protruding to the bottom-right or top-left
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymax = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmin = cutx

                # Add the adjusted bounding box to the temporary list
                tmp_box.append(xmin)
                tmp_box.append(ymin)
                tmp_box.append(xmax)
                tmp_box.append(ymax)
                tmp_box.append(bbox[4])

                # Add the temporary list to the merge list
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_mosaic_data(self, image_data_list, label_list):
        """
        Apply mosaic augmentation to the image data and labels.

        Args:
            image_data_list (list): List of image data.
            label_list (list): List of label data.

        Returns:
            tuple: Tuple containing the mosaic image data and bounding boxes.
        """

        bboxes = []

        cutx = int(self.width * self.rand(0.3, 0.7))
        cuty = int(self.height * self.rand(0.3, 0.7))

        # Calculate the placement position for each image
        place_x = [0, 0, cutx, cutx]
        place_y = [0, cuty, cuty, 0]

        # Create the mosaic image
        mosaic_image = Image.new("RGB", (self.width, self.height))

        for n, image in enumerate(image_data_list):
            bbox = np.zeros_like(label_list[n])
            image_w, image_h = image.size

            # Convert cx,cy,w,h -> xmin,ymin,xmax,ymax
            bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3] = cxcywh2xyxy(
                label_list[n][:, 0],
                label_list[n][:, 1],
                label_list[n][:, 2],
                label_list[n][:, 3],
            )
            bbox[:, 4] = label_list[n][:, 4]

            # Flipping image horizontally
            flip = self.rand() < 0.5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                bbox[:, [0, 2]] = image_w - bbox[:, [2, 0]]

            new_ar = (image_w / image_h) * (self.rand(0.7, 1.3) / self.rand(0.7, 1.3))

            # Scale value for resizing the image
            scale = self.rand(0.4, 1)

            # Calculate the image size based on the new aspect ratio
            if new_ar < 1:  # Vertical image
                nh = int(scale * self.height)
                nw = int(nh * new_ar)
            else:  # Horizontal image
                nw = int(scale * self.width)
                nh = int(nw * (1 / new_ar))

            if (self.width * self.height) < (image_w * image_h):
                resize_image = image.resize((nw, nh), resample=Image.LANCZOS)
            else:
                resize_image = image.resize((nw, nh), resample=Image.BICUBIC)

            # Get the placement position for the image
            dx = place_x[n]
            dy = place_y[n]

            # Adjust the size if it exceeds the original image when pasting (horizontal direction)
            if (dx + nw) > self.width:
                dst_width = self.width - dx
            else:
                dst_width = nw

            # Adjust the size if it exceeds the original image when pasting (vertical direction)
            if (dy + nh) > self.height:
                dst_height = self.height - dy
            else:
                dst_height = nh

            # Place the image
            mosaic_image.paste(resize_image.crop((0, 0, dst_width, dst_height)), (dx, dy))
            resize_image = None

            # Update the bounding box coordinates
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * (nw / image_w) + dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * (nh / image_h) + dy

            # Select valid bounding boxes
            box_w = bbox[:, 2] - bbox[:, 0]
            box_h = bbox[:, 3] - bbox[:, 1]
            bbox = bbox[np.logical_and(box_w > 1, box_h > 1)]

            bboxes.append(bbox)

        bboxes = self.merge_bboxes(bboxes, cutx, cuty)
        bboxes = np.array(bboxes, dtype=np.float32)

        # Randomly change the hue, saturation, and value
        hue_shift = np.random.uniform(-0.1, 0.1)
        sat_shift = np.random.uniform(-0.7, 0.7)
        val_shift = np.random.uniform(-0.4, 0.4)

        # Convert the image to HSV color space and split the channels
        mosaic_image = mosaic_image.convert("HSV")
        hue, sat, val = mosaic_image.split()

        # Apply color shifts
        hue = hue.point(lambda x: ((x + int(hue_shift * 255)) % 256))
        sat = sat.point(lambda x: np.clip(x + int(sat_shift * 255), 0, 255))
        val = val.point(lambda x: np.clip(x + int(val_shift * 255), 0, 255))

        mosaic_image = Image.merge("HSV", (hue, sat, val)).convert("RGB")

        # Convert xmin,ymin,xmax,ymax -> cx,cy,w,h
        mosaic_boxes = np.zeros((len(bboxes), 5), np.float32)
        mosaic_boxes[:, 0], mosaic_boxes[:, 1], mosaic_boxes[:, 2], mosaic_boxes[:, 3] = xyxy2cxcywh(
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 2],
            bboxes[:, 3],
        )
        mosaic_boxes[:, 4] = bboxes[:, 4]

        return np.array(mosaic_image, dtype=np.float32), mosaic_boxes

    def pick_up_sample_data(self, sample_num, idx):
        """
        Pick up sample data for mosaic augmentation.

        Args:
            sample_num (int): Number of samples to pick up.
            idx (int): Index of the current data.

        Returns:
            tuple: Tuple containing the picked up image data and labels.
        """

        data_index_list = list(range((len(self.image_path_list))))
        data_index_list.pop(idx)
        sample_index_list = sample(data_index_list, sample_num)
        image_data_list = []
        label_list = []
        for data_idx in sample_index_list:
            image_path = self.image_path_list[data_idx]
            image_data = imread(image_path)
            object_data = self.label_list[data_idx]
            image_data_list.append(image_data)
            label_list.append(object_data)

        return image_data_list, label_list

    def paste_image_at_random_position(self, src_image, dst_image):
        """
         Paste the source image onto the destination image at a random position.

         Args:
            src_image (PIL.Image): Source image to paste.
            dst_image (PIL.Image): Destination image to paste onto.

        Returns:
            tuple: Tuple containing the pasted coordinates (x, y).
        """

        if self.training:
            pb = np.random.rand()
        else:
            pb = 0

        # Calculate the position to paste the image
        dst_dx = int(pb * (dst_image.size[0] - src_image.size[0]))
        dst_dy = int(pb * (dst_image.size[1] - src_image.size[1]))

        # Set image
        dst_image.paste(src_image, (dst_dx, dst_dy))

        return dst_dx, dst_dy

    def scale_image_with_aspect_ratio(self, image_data, label):
        """
        Scale the image while maintaining the aspect ratio.

        Args:
            image_data (PIL.Image): Image data to scale.
            label (np.ndarray): Label data.

        Returns:
            tuple: Tuple containing the scaled image data and updated label.
        """

        # Initialize
        dst_label = np.zeros((label.shape[0], label.shape[1]), dtype=label.dtype)
        dst_image_data = Image.new("RGB", (self.width, self.height), color=(0, 0, 0))

        # Resize image data
        resize_width_rate = self.width / image_data.width
        resize_height_rate = self.height / image_data.height
        resize_rate = min(resize_width_rate, resize_height_rate)
        resize_width = int(image_data.width * resize_rate)
        resize_height = int(image_data.height * resize_rate)

        # Image downscaling if (self.width * self.height) < (image_data.width * image_data.height)
        if (self.width * self.height) < (image_data.width * image_data.height):
            tmp_image_data = image_data.resize((resize_width, resize_height), resample=Image.LANCZOS)
        # Image upscaling
        else:
            tmp_image_data = image_data.resize((resize_width, resize_height), resample=Image.BICUBIC)

        # Resize object data
        dst_label[:, 0] = label[:, 0] * resize_rate
        dst_label[:, 1] = label[:, 1] * resize_rate
        dst_label[:, 2] = label[:, 2] * resize_rate
        dst_label[:, 3] = label[:, 3] * resize_rate
        dst_label[:, 4] = label[:, 4]

        # Pasting the resized image onto a black image
        dst_dx, dst_dy = self.paste_image_at_random_position(tmp_image_data, dst_image_data)
        tmp_image_data = None

        # Updating object coordinates
        dst_label[:, 0] = dst_label[:, 0] + dst_dx
        dst_label[:, 1] = dst_label[:, 1] + dst_dy

        return np.array(dst_image_data, dtype=np.float32), dst_label

    def correct_protruding_coordinates(self, image_shape, label):
        """
        Correct coordinates that protrude outside the image.

        Args:
            image_shape (tuple): Shape of the image (width, height).
            label (np.ndarray): Label data.

        Returns:
            np.ndarray: Corrected label data.
        """

        image_w, image_h = image_shape[:2]
        bbox_list = []

        for bbox in label:
            bbox_ltx, bbox_lty, bbox_rbx, bbox_rby = cxcywh2xyxy(bbox[0], bbox[1], bbox[2], bbox[3])

            bbox_ltx = max(0, bbox_ltx)
            bbox_lty = max(0, bbox_lty)
            bbox_rbx = min(image_w, bbox_rbx)
            bbox_rby = min(image_h, bbox_rby)

            bbox_cx, bbox_cy, bbox_width, bbox_height = xyxy2cxcywh(bbox_ltx, bbox_lty, bbox_rbx, bbox_rby)
            bbox_list.append([bbox_cx, bbox_cy, bbox_width, bbox_height, bbox[4]])

        bboxes = np.array(bbox_list, dtype=np.float32)
        return bboxes

    def rand(self, min_value=0, max_value=1):
        """
        Generate a random float value between min_value and max_value.

        Args:
            min_value (float): Minimum value (default: 0).
            max_value (float): Maximum value (default: 1).

        Returns:
            float: Random float value.
        """

        return np.random.rand() * (max_value - min_value) + min_value
