from abc import ABC, abstractmethod
from typing import Type, Union

import numpy as np
import torchvision.transforms as torch_vis_transforms
from PIL import Image, ImageFilter
from augly.image import utils as imutils
from augly.image.transforms import BaseTransform

_DEBUG = False


class BaseTransf(ABC):
    def __call__(self, *args, **kwargs):
        if _DEBUG:
            print(self.__class__.__qualname__)

        img = self.custom_transf(*args, **kwargs)
        return img

    @abstractmethod
    def custom_transf(self, *args, **kwargs):
        raise NotImplementedError


class FromNp0255ToPILImage(BaseTransf):
    def custom_transf(self, img) -> Image.Image:
        img = Image.fromarray(img)
        return img


class FromPILImageToNp0255(BaseTransf):
    def custom_transf(self, img) -> Image.Image:
        img = np.uint8(img)
        return img


class Convert2RGB(BaseTransf):
    def custom_transf(self, img) -> Image.Image:
        return img.convert('RGB')


class NormalizeMinMax(BaseTransf):
    def custom_transf(self, img) -> Image.Image:
        imin = img.min()
        imax = img.max()
        if imax != 0:
            img = (img - imin) / (imax - imin)
        return img


class NormalizeMask0255(BaseTransf):
    def custom_transf(self, masks) -> Image.Image:
        imin = masks.min()
        imax = masks.max()
        if imax != 0:
            masks = 255 * ((masks - imin) / (imax - imin))
        return masks


class RandomEdgeEnhance(BaseTransform):
    def __init__(self, mode: Union[Type[ImageFilter.EDGE_ENHANCE],
                                   Type[ImageFilter.EDGE_ENHANCE_MORE]] = ImageFilter.EDGE_ENHANCE,
                 p: float = 1.0, ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, image: Image.Image, *args) -> Image.Image:
        if _DEBUG:
            print(self.__class__.__qualname__)
        return image.convert('RGB').filter(self.mode).convert('RGB')


class ColorJitter(BaseTransform):
    def __init__(self, values_color_jitter):
        super().__init__()
        self.trans = torch_vis_transforms.ColorJitter(**values_color_jitter)

    def apply_transform(self, image: Image.Image, *args) -> Image.Image:
        if _DEBUG:
            print(self.__class__.__qualname__)

        image = image.convert('RGB')
        src_mode = image.mode
        image = self.trans(image.convert('RGB'))
        imutils.get_metadata(metadata=None, function_name=self.__class__.__qualname__)
        return imutils.ret_and_save_image(image, None, src_mode)
