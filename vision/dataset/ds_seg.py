#!/usr/bin/env python
import gc
import inspect
import json
import logging
import os
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from path import Path
from torch import Tensor
from torch.utils.data import Dataset

from vision.dataset.build_transf import build_transforms, build_transforms_val
from vision.utils.image.io import img_read_PIL_RGB


class DatasetSeg(Dataset):
    def __init__(self,
                 stage: str,

                 n_classes: int,
                 classes: List[str],
                 input_size: int,
                 files_images: Dict[str, List[Union[int, str]]],
                 files_mask: Dict[str, Dict[str, str]],

                 transf_degree: float = 1.,
                 rank: int = 0,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.stage = stage
        self.n_classes = n_classes
        self.classes = classes
        self.rank = rank
        self.verbose = verbose

        n_imgs = 0
        self.images, self.idx_to_img_idx = [], []
        self.masks = {class_name: [] for class_name in self.classes}
        for fn_annot_name, (repeat, fn_images) in files_images.items():
            images = json.load(open(os.path.expanduser(fn_images)))
            self.logger.info(f'len repeated {fn_annot_name:<25} = {repeat * len(images)}')
            for irep in range(repeat):
                self.idx_to_img_idx.extend(list(len(self.images) + np.arange(len(images))))
            n_imgs += repeat * len(images)
            self.images += images
            for class_name, fn_mask_cls in files_mask[fn_annot_name].items():
                if fn_mask_cls is None:
                    self.masks[class_name] += len(images) * [None]
                elif fn_mask_cls == "zero":
                    self.masks[class_name] += len(images) * ["zero"]
                else:
                    self.masks[class_name] += json.load(open(os.path.expanduser(fn_mask_cls)))

        preprocessor = dict(
            mean=[0, 0, 0], std=[1, 1, 1]  # normalization in the model (input in range [0, 1])
        )

        if stage == "train":
            self.transforms = build_transforms(
                input_size, input_size, preprocessor, transf_degree, norm_minmax=False)
        elif "test" in stage:
            self.transforms = build_transforms_val(input_size, input_size, preprocessor, norm_minmax=False)
        elif "valid" in stage:
            self.transforms = build_transforms_val(input_size, input_size, preprocessor, norm_minmax=False)
        else:
            raise ValueError(f"Not implemented transform for stage = {stage}")

        # Clean and save.
        gc.collect()

        self.logger.info(f'{45 * "-"}')
        self.logger.info(f'len dataset repeated {"all":<16}  = {len(self.idx_to_img_idx)}')
        self.logger.info(f'len dataset  {"all":<25} = {len(self.images)}')
        self.logger.info(f'{self.__class__.__qualname__} initiated')

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.idx_to_img_idx)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, str]:

        idx_img = self.idx_to_img_idx[idx]

        img = img_read_PIL_RGB(self.images[idx_img])
        img = np.uint8(img)

        masks = np.zeros((img.shape[0], img.shape[1], self.n_classes), dtype=np.uint8)
        consider = np.zeros(self.n_classes, dtype=bool)
        for icls, class_name in enumerate(self.classes):
            fn = self.masks[class_name][idx_img]
            if fn is not None and fn != "zero":
                tmp = img_read_PIL_RGB(fn)
                masks[:, :, icls] = np.uint8(tmp)[:, :, 0]
                consider[icls] = True
            if fn == "zero":
                consider[icls] = True

        img, masks, _ = self.transforms(img, masks)

        return img, masks, torch.tensor(consider), Path(self.images[idx_img]).stem

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor, Tensor, List[str]]:
        s_image, s_mask, s_consider, s_fn = [], [], [], []
        for image, mask, consider, fn in batch:
            s_image.append(image)
            s_mask.append(mask)
            s_consider.append(consider)
            s_fn.append(fn)
        s_image = torch.stack(s_image, dim=0)
        s_mask = torch.stack(s_mask, dim=0)
        s_consider = torch.stack(s_consider, dim=0)
        return s_image, s_mask, s_consider, s_fn
