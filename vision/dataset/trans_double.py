#!/usr/bin/env python
import inspect
import logging
import math
import random
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
import torchvision.transforms as torch_vis_transforms
from PIL import Image
import torch.nn.functional as functional

from vision.dataset.geometry import RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, rotate_no_crop
from vision.dataset.torch_vis_mod import RandomResizedCrop
from vision.dataset.utils import check_same_hw_np


def no_255_to_zero(mask):
    mask[mask < 128] = 0
    mask[mask >= 128] = 255
    return mask


class RandomResizedCropImgMask:
    def __init__(self, input_size_h, input_size_w, scale):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.out_size = (input_size_h, input_size_w)
        self.scale = scale

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        width_orig, height_orig = img.shape[:2]
        mat = np.eye(3)

        img = torch.tensor(img).permute(2, 0, 1)
        img, i, j, h, w = RandomResizedCrop(self.out_size, scale=self.scale)(img)
        img = img.permute(1, 2, 0)
        img = np.uint8(img)

        # ResizedCrop mask
        masks = masks[i:i + h, j:j + w, :]
        masks = torch.tensor(masks.copy()).permute(2, 0, 1)
        masks = torch_vis_transforms.Resize(self.out_size, antialias=None)(masks).permute(1, 2, 0)
        masks = np.uint8(masks)
        masks = no_255_to_zero(masks)

        # Crop => Translation
        transform_translate = np.eye(3)
        transform_translate[0, 2] = +(width_orig / 2 - (j + w / 2))
        transform_translate[1, 2] = +(height_orig / 2 - (i + h / 2))

        # ResizedCrop => Scale
        scale_w = self.out_size[1] / w
        scale_h = self.out_size[0] / h
        transform_scale = np.asarray([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]])

        # Final trans mat
        mat = transform_scale @ transform_translate @ mat

        return img, masks, mat


class SquareImgMask:
    def __init__(self):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        current_height, current_width = img.shape[-2], img.shape[-1]
        paddded_size = max(current_height, current_width)

        pad_height = max(0, paddded_size - current_height)
        pad_width = max(0, paddded_size - current_width)

        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), constant_values=0, mode='constant')
        masks = np.pad(masks, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), constant_values=0, mode='constant')

        return img, masks, np.eye(3)


class ResizeImgMask:
    def __init__(self, h_out_size, w_out_size):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.h_out_size = h_out_size
        self.w_out_size = w_out_size
        self.resize = torch_vis_transforms.Resize((self.h_out_size, self.w_out_size), antialias=None)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        height_orig, width_orig = img.shape[-2], img.shape[-1]
        mat = np.eye(3)

        img = torch.tensor(img)
        masks = torch.tensor(masks)
        img = self.resize(img.permute(2, 0, 1)).permute(1, 2, 0)
        masks = self.resize(masks.permute(2, 0, 1)).permute(1, 2, 0)
        img = np.uint8(img)
        masks = np.uint8(masks)
        masks = no_255_to_zero(masks)

        # Crop => Translation
        transform_translate = np.eye(3)

        # ResizedCrop => Scale
        scale_w = self.w_out_size / width_orig
        scale_h = self.h_out_size / height_orig
        transform_scale = np.asarray([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]])

        # Final trans mat
        mat = transform_scale @ transform_translate @ mat

        return img, masks, mat


class RandomVerticalFlipImgMask:
    def __init__(self, p: float = 0.5):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.trans = RandomVerticalFlip(p=p)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        img, mat = self.trans(img)
        if np.any(mat != np.eye(3)):
            masks = np.flip(masks, axis=0)
        return img, masks, mat


class RandomHorizontalFlipImgMask:
    def __init__(self, p: float = 0.5):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.trans = RandomHorizontalFlip(p=p)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)
        
        img, mat = self.trans(img)
        if np.any(mat != np.eye(3)):
            masks = np.flip(masks, axis=1)
        return img, masks, mat


# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotatedRectWithMaxArea(w, h, angle):
    """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


class RandomRotationNoCrop:
    def __init__(self,
                 interpolation=cv2.INTER_LINEAR,
                 fillcolor=0):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    def __call__(self, img: np.ndarray, masks: np.ndarray, degrees: float = 0):
        check_same_hw_np(self.logger, img, masks)

        img = rotate_no_crop(img,
                             degrees,
                             interpolation=self.interpolation,
                             fillcolor=self.fillcolor)

        masks = rotate_no_crop(masks,
                              degrees,
                              interpolation=self.interpolation,
                              fillcolor=0)
        masks = no_255_to_zero(masks)

        radians = math.radians(degrees)
        transform_rotate = np.asarray([
            [math.cos(radians), math.sin(radians), 0],
            [-math.sin(radians), math.cos(radians), 0],
            [0, 0, 1]
        ])

        return img, masks, transform_rotate


class RandomRotationInnerImgMask:
    def __init__(self, degrees=(0, 360), fillcolor: int = 0):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.degrees = degrees
        self.trans = RandomRotationNoCrop(fillcolor=fillcolor)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        h_orig, w_orig = img.shape[:2]

        # Rotate
        degrees = random.uniform(self.degrees[0], self.degrees[1])
        img, masks, mat = self.trans(img, masks, degrees)
        masks = no_255_to_zero(masks)

        wr, hr = rotatedRectWithMaxArea(w_orig, h_orig, math.radians(degrees))
        h_rot, w_rot = img.shape[:2]
        h0 = int((h_rot - 1) / 2 - (hr - 1) / 2)
        h1 = int((h_rot - 1) / 2 + (hr - 1) / 2)
        w0 = int((w_rot - 1) / 2 - (wr - 1) / 2)
        w1 = int((w_rot - 1) / 2 + (wr - 1) / 2)

        img = img[h0:h1, w0:w1, :]
        masks = masks[h0:h1, w0:w1, :]

        return img, masks, mat


class RandomShearInnerMultiple:
    def __init__(self, shear: float = 0.0):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.trans = RandomAffine(degrees=0, shear=shear, fillcolor=0)

    def __call__(self, img: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_same_hw_np(self.logger, img, masks)

        shear_x_or_y = np.random.random() > 0.5
        if shear_x_or_y:
            img = img.transpose(1, 0, 2)
            masks = masks.transpose(1, 0, 2)

        img, mask, mat = self.trans(img, masks)
        masks = no_255_to_zero(masks)

        if shear_x_or_y:
            img = img.transpose(1, 0, 2)
            masks = masks.transpose(1, 0, 2)
            mat = np.asarray([
                [mat[1, 1], mat[1, 0], 0],
                [mat[0, 1], mat[0, 0], 0],
                [0, 0, 1]
            ])

        return img, masks, mat
