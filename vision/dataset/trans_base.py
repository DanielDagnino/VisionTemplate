import augly.image as augly_image
import numpy as np
import torch
import torchvision.transforms as torch_vis_transforms

from vision.dataset import transf_single
from PIL import Image


class BaseTransMask:
    def __init__(self, trans, **kwargs):
        self.trans = trans(**kwargs)

    def __call__(self, img, mask):
        return img, self.trans(mask), np.eye(3)


class BaseTransImg:
    def __init__(self, trans, **kwargs):
        self.trans = trans(**kwargs)

    def __call__(self, img, mask):
        return self.trans(img), mask, np.eye(3)


class BaseTransImgMask:
    def __init__(self, trans, **kwargs):
        self.trans = trans(**kwargs)

    def __call__(self, img, mask):
        return self.trans(img), self.trans(mask), np.eye(3)


class ToTensor(BaseTransImgMask):
    def __init__(self):
        super().__init__(torch_vis_transforms.ToTensor)


class ToTensorImg(BaseTransImg):
    def __init__(self):
        super().__init__(torch_vis_transforms.ToTensor)


class PILToTensor(BaseTransImgMask):
    def __init__(self):
        super().__init__(torch_vis_transforms.PILToTensor)


class PILToTensorImg(BaseTransImg):
    def __init__(self):
        super().__init__(torch_vis_transforms.PILToTensor)

    def __call__(self, img, mask):
        img, mask, mat = super().__call__(img, mask)
        return img.to(torch.float), mask, mat


class Convert2RGB(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(transf_single.Convert2RGB, **kwargs)


class Normalize(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(torch_vis_transforms.Normalize, **kwargs)


class FromNp0255ToPILImage(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(transf_single.FromNp0255ToPILImage, **kwargs)


class FromPILImageToNp0255(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(transf_single.FromPILImageToNp0255, **kwargs)


class NormalizeMinMax(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(transf_single.NormalizeMinMax, **kwargs)


class NormalizeMask0255(BaseTransMask):
    def __init__(self, **kwargs):
        super().__init__(transf_single.NormalizeMask0255, **kwargs)


class RandomPixelization(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(augly_image.RandomPixelization, **kwargs)


class RandomBlur(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(augly_image.RandomBlur, **kwargs)


class EncodingQuality(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(augly_image.EncodingQuality, **kwargs)


class RandomEdgeEnhance(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(transf_single.RandomEdgeEnhance, **kwargs)


class ColorJitter(BaseTransImg):
    def __init__(self, **kwargs):
        super().__init__(torch_vis_transforms.ColorJitter, **kwargs)
