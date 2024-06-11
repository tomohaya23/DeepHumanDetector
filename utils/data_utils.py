#!/usr/bin/env python
# coding:utf-8

import numpy as np
from PIL import Image


def imread(image_path: str) -> np.ndarray:
    """
    Read an image file using Pillow and return it as a 3-channel numpy array in RGB order.
    :param image_path: Path to the image file to read
    :return: Numpy array of the read image
    """
    try:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        return image_rgb
    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}. Error: {str(e)}")


def cxcywh2xyxy(cx: float, cy: float, width: float, height: float) -> tuple:
    """
    Convert center coordinates (cx, cy), width, and height to
    top-left coordinates (xmin, ymin) and bottom-right coordinates (xmax, ymax).
    :param cx: x-coordinate of the center
    :param cy: y-coordinate of the center
    :param width: Width
    :param height: Height
    :return: Tuple of (xmin, ymin, xmax, ymax)
    """
    xmin = cx - width / 2
    ymin = cy - height / 2
    xmax = xmin + width
    ymax = ymin + height
    return xmin, ymin, xmax, ymax


def xyxy2cxcywh(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple:
    """
    Convert top-left coordinates (xmin, ymin) and bottom-right coordinates (xmax, ymax) to
    center coordinates (cx, cy), width, and height.
    :param xmin: x-coordinate of the top-left corner
    :param ymin: y-coordinate of the top-left corner
    :param xmax: x-coordinate of the bottom-right corner
    :param ymax: y-coordinate of the bottom-right corner
    :return: Tuple of (cx, cy, width, height)
    """
    width = xmax - xmin
    height = ymax - ymin
    cx = xmin + width / 2
    cy = ymin + height / 2
    return cx, cy, width, height
