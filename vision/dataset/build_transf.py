from typing import Callable

import torchvision.transforms as torch_vis_transforms
from PIL import ImageFilter

from vision.dataset import transf_single
from vision.dataset.trans_base import ToTensor, FromNp0255ToPILImage, FromPILImageToNp0255, \
    RandomPixelization, RandomBlur, EncodingQuality, \
    RandomEdgeEnhance, ColorJitter, \
    NormalizeMinMax, Normalize
from vision.dataset.trans_double import ResizeImgMask, RandomResizedCropImgMask, SquareImgMask, \
    RandomRotationInnerImgMask, RandomShearInnerMultiple, \
    RandomHorizontalFlipImgMask, RandomVerticalFlipImgMask
from vision.dataset.utils import ShuffledAugMultiple, OneOfMultiple, SequentialAugMultiple, SequentialAug


_DEBUG = False


def build_preprocess(input_size_h: int, input_size_w: int,
                     preprocessor: dict, norm_minmax: bool) -> Callable:
    _preprocess = [
        torch_vis_transforms.Resize((input_size_h, input_size_w))
    ]
    _preprocess += [
        torch_vis_transforms.ToTensor(),
        Normalize(mean=preprocessor['mean'], std=preprocessor['std'])
        if not norm_minmax else transf_single.NormalizeMinMax()
    ]

    return SequentialAug(_preprocess)


def build_transforms_val(
        input_size_h: int, input_size_w: int, preprocessor, norm_minmax: bool = False,
) -> Callable:
    aug = [
        SquareImgMask(),
        ResizeImgMask(input_size_h, input_size_w),
    ]
    aug += [
        ToTensor(),
        Normalize(mean=preprocessor['mean'], std=preprocessor['std']) if not norm_minmax else NormalizeMinMax()
    ]
    return SequentialAugMultiple(aug)


def build_transforms(
        input_size_h: int, input_size_w: int, preprocessor, transf_degree: float = 1., norm_minmax: bool = False,
) -> Callable:
    aug_pixels = [
        OneOfMultiple([
            RandomPixelization(min_ratio=0.20*transf_degree, p=0.50*transf_degree if not _DEBUG else 1),
            RandomBlur(max_radius=3*transf_degree, p=0.50*transf_degree if not _DEBUG else 1),
        ]),
        OneOfMultiple([EncodingQuality(quality=q) for q in [50, 70]], p=0.50*transf_degree if not _DEBUG else 1),
    ]

    aug_filter = [
        OneOfMultiple([
            ColorJitter(
                brightness=0.40*transf_degree, contrast=0.40*transf_degree, saturation=0.40*transf_degree,
                hue=(-0.10*transf_degree, 0.10*transf_degree))
        ], p=0.60*transf_degree if not _DEBUG else 1),
    ]

    aug_geo = [
        OneOfMultiple([
            RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE),
        ], p=0.20*transf_degree if not _DEBUG else 1),
    ]

    aug = [
        SquareImgMask(),
        ResizeImgMask(input_size_h, input_size_w),

        RandomHorizontalFlipImgMask(),
        RandomVerticalFlipImgMask(),
        OneOfMultiple([RandomRotationInnerImgMask(degrees=(-180*transf_degree, 180*transf_degree))],
                      p=0.80*transf_degree if not _DEBUG else 1),
        OneOfMultiple([RandomShearInnerMultiple(shear=20*transf_degree)], p=0.70*transf_degree if not _DEBUG else 1),
        RandomResizedCropImgMask(input_size_h, input_size_w, scale=(min(1., 0.80*(1+transf_degree)), 1.)),

        FromNp0255ToPILImage(),
        ShuffledAugMultiple(aug_pixels + aug_filter + aug_geo),
        FromPILImageToNp0255(),

        ResizeImgMask(input_size_h, input_size_w),
    ]
    aug += [
        ToTensor(),
        Normalize(mean=preprocessor['mean'], std=preprocessor['std']) if not norm_minmax else NormalizeMinMax()
    ]

    return SequentialAugMultiple(aug)
