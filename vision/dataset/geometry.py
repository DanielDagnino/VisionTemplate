import math
import numbers
import random
from typing import Tuple

import cv2
import numpy as np


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.randint(0, 1) < self.p:
            o_img = cv2.flip(img, 1)
            t = np.eye(3)
            t[0, 0] = -1
            return o_img, t
        else:
            return img, np.eye(3)


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.randint(0, 1) < self.p:
            o_img = cv2.flip(img, 0)
            t = np.eye(3)
            t[1, 1] = -1
            return o_img, t
        else:
            return img, np.eye(3)


def affine(
        img: np.ndarray,
        angle: float,
        translate: Tuple[float, float],
        scale: float,
        shear: float,
        interpolation=cv2.INTER_LINEAR,
        mode=cv2.BORDER_CONSTANT,
        fillcolor=0
):
    def gen_affine_matrix(
            center: Tuple[float, float], angle: float, translate: Tuple[float, float], scale: float, shear: float
    ) -> np.ndarray:
        angle = math.radians(angle)
        shear = math.radians(shear)
        transl = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
        move_center = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
        rot_scale_shear = np.array([[math.cos(angle) * scale, -math.sin(angle + shear) * scale, 0],
                                    [math.sin(angle) * scale, math.cos(angle + shear) * scale, 0], [0, 0, 1]])
        matrix = transl @ move_center @ rot_scale_shear @ np.linalg.inv(move_center)
        return matrix[:2, :]

    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
    matrix = gen_affine_matrix(center, angle, translate, scale, shear)

    if img.shape[2] == 1:
        return cv2.warpAffine(
            img, matrix, output_size[::-1], interpolation, borderMode=mode, borderValue=fillcolor)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(
            img, matrix, output_size[::-1], interpolation, borderMode=mode, borderValue=fillcolor)


class RandomAffine:
    """
    https://github.com/jbohnslav/opencv_transforms/blob/fd91e4987a6929be9334b40f0f809d7a2709383f/opencv_transforms/transforms.py#L862

    Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        fillcolor (int): Optional fill color for the area outside the transform in the output image.
    """

    def __init__(self,
                 degrees: float,
                 translate=None,
                 scale: float = None,
                 shear: float = None,
                 interpolation=cv2.INTER_LINEAR,
                 fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.interpolation = interpolation
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        translations = (0, 0)
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))

        scale = 1.0
        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])

        shear = 0.0
        if shears is not None:
            shear = random.uniform(shears[0], shears[1])

        return angle, translations, scale, shear

    def __call__(self, img, masks):
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale,
                                                            self.shear, (img.shape[1], img.shape[0]))

        o_img = affine(img,
                       angle, translations, scale, shear,
                       interpolation=self.interpolation,
                       fillcolor=self.fillcolor)
        # Transform matrix
        transform_rotate = np.asarray([
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
            [0, 0, 1]
        ])

        transform_shear = np.eye(3)
        transform_shear[0, 1] = -math.sin(math.radians(shear))
        transform_shear[1, 1] = math.cos(math.radians(shear))

        transform_scale = np.asarray([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

        transform_translate = np.eye(3)
        transform_translate[0, 2] = translations[0] / img.shape[0]
        transform_translate[1, 2] = translations[1] / img.shape[1]

        transform = transform_rotate @ transform_translate @ transform_scale @ transform_shear

        masks = affine(masks,
                      angle, translations, scale, shear,
                      interpolation=self.interpolation,
                      fillcolor=0)
        return o_img, masks, transform


# https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
def rotate_no_crop(img, angle, interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_CONSTANT, fillcolor=0):

    height, width = img.shape[0:2]
    center = (width * 0.5 - 0.5, height * 0.5 - 0.5)

    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(math.cos(math.radians(angle)))
    abs_sin = abs(math.sin(math.radians(angle)))

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]

    if img.shape[2] == 1:
        return cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), interpolation, borderMode=mode,
                              borderValue=fillcolor)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), interpolation, borderMode=mode,
                              borderValue=fillcolor)
