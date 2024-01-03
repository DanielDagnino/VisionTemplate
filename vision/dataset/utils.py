import random
from typing import Optional, List, Union

import numpy as np
from PIL import Image
from torch import Tensor


def check_same_hw_pil(logger, img: Image.Image, masks: List[Image.Image]):
    for mask in masks:
        msg = ''
        try:
            if img.size != mask.size:
                msg = f'Image and Mask must have the same H and W, however: \n' \
                      f'img.size = {img.size} \n' \
                      f'mask.size = {mask.size} \n'
        except Exception as excpt:
            msg = f'Image and Mask must be PIL images with the same size, however: \n' \
                  f'type(img) = {type(img)} \n' \
                  f'type(mask) = {type(mask)} \n' \
                  f'img.size = {img.size} \n' \
                  f'mask.size = {mask.size} \n'
            msg += excpt
        if msg:
            logger.error(msg)
            raise ValueError(msg)


def check_same_hw_tensor(logger, img: Union[Tensor, np.ndarray], masks: Union[Tensor, np.ndarray]):
    msg = ''
    try:
        if img.shape[1:] != masks.shape[1:]:
            msg = f'Image and Mask must have the same H and W, however: \n' \
                  f'img.size = {img.shape} \n' \
                  f'mask.size = {masks.shape} \n'
    except Exception as excpt:
        msg = f'Image and Mask must be Tensor images with the same size, however: \n' \
              f'type(img) = {type(img)} \n' \
              f'type(mask) = {type(masks)} \n' \
              f'img.shape = {img.shape} \n' \
              f'mask.shape = {masks.shape} \n'
        msg += excpt
    if msg:
        logger.error(msg)
        raise ValueError(msg)


def check_same_hw_np(logger, img: Union[Tensor, np.ndarray], masks: Union[Tensor, np.ndarray]):
    msg = ''
    try:
        if img.shape[:2] != masks.shape[:2]:
            msg = f'Image and Mask must have the same H and W, however: \n' \
                  f'img.size = {img.shape} \n' \
                  f'mask.size = {masks.shape} \n'
    except Exception as excpt:
        msg = f'Image and Mask must be Tensor images with the same size, however: \n' \
              f'type(img) = {type(img)} \n' \
              f'type(mask) = {type(masks)} \n' \
              f'img.shape = {img.shape} \n' \
              f'mask.shape = {masks.shape} \n'
        msg += excpt
    if msg:
        logger.error(msg)
        raise ValueError(msg)


class OneOfMultiple:
    def __init__(self, transforms, p=1):
        self.transforms = transforms
        self.p = p

    def __call__(self, x, mask):
        if random.random() > self.p:
            return x, mask, np.eye(3)

        trans = random.choice(self.transforms)
        x, mask, mat = trans(x, mask)
        return x, mask, mat


class RandomApplyMultiple:
    def __init__(self, transforms, p=1):
        self.transforms = transforms
        self.p = p

    def __call__(self, x, mask):
        if random.random() > self.p:
            return x, mask, np.eye(3)

        mat = None
        for trans in self.transforms:
            x, mask, mat_tmp = trans(x, mask)
            if mat is None:
                mat = mat_tmp
            else:
                mat = mat_tmp @ mat
        return x, mask, mat


class ShuffledAugMultiple:
    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, x, mask):
        mat = None
        shuffled_aug_list = random.sample(self.aug_list, len(self.aug_list))
        for op in shuffled_aug_list:
            x, mask, mat_tmp = op(x, mask)
            if mat is None:
                mat = mat_tmp
            else:
                mat = mat_tmp @ mat
        return x, mask, mat


class SequentialAug:
    def __init__(self, aug_list: list):
        self.aug_list = aug_list

    def __call__(self, x):
        for op in self.aug_list:
            x = op(x)
        return x


class SequentialAugMultiple:
    def __init__(self, aug_list: list):
        self.aug_list = aug_list

    def __call__(self, x, mask):
        mat = None
        for op in self.aug_list:
            x, mask, mat_tmp = op(x, mask)
            if mat is None:
                mat = mat_tmp
            else:
                mat = mat_tmp @ mat
        return x, mask, mat


def torch_to_np_image(img: Tensor, ibatch: Optional[int] = None, is_01=False):
    if ibatch is not None:
        img = img[ibatch]
    img = img.permute(1, 2, 0)
    if not is_01:
        if img.max() != img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img[...] = 0
    img = 255 * img
    img = np.array(img, dtype=np.uint8)
    return img
