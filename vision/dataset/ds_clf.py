#!/usr/bin/env python
import gc
import inspect
import json
import logging
import os
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from path import Path
from torch import Tensor
from torch.utils.data import Dataset

from vision.dataset.build_transf import build_transforms, build_transforms_val
from vision.utils.image.io import img_read_PIL_RGB


class DatasetClf(Dataset):
    def __init__(self,
                 stage: str,
                 n_classes: int,
                 input_size: int,
                 base_img_dirs: List[str],
                 fns_annots: List[Tuple[int, int, str]],
                 transf_degree: float = 1.,
                 rank: int = 0,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.stage = stage
        self.n_classes = n_classes
        self.rank = rank
        self.verbose = verbose

        base_img_dirs = [Path(os.path.expanduser(_)) for _ in base_img_dirs]
        fns_annots = [[repeat, reduce, Path(os.path.expanduser(fn_annots))]
                      if fn_annots is not None else [repeat, reduce, None]
                      for repeat, reduce, fn_annots in fns_annots]
        self.data: List[List[Dict[str, Any]]] = []
        for base_img_dir, [repeat, reduce, fn_annots] in zip(base_img_dirs, fns_annots):
            if fn_annots is not None:
                data = []
                sample_accum = []
                for sample in json.load(open(os.path.expanduser(fn_annots))):
                    sample["path"] = base_img_dir / sample["img_name"]
                    sample_accum.append(sample)
                    if len(sample_accum) == reduce:
                        data.append(sample_accum)
                        sample_accum = []
                if sample_accum != []:
                    data.append(sample_accum)
                self.data.extend(repeat * data)
            else:
                data = []
                sample_accum = []
                for fn in base_img_dir.walkfiles():
                    no_annotation = {
                        "path": fn,
                        "img_name": None,
                        "annotator_name": None,
                        "annotations": self.n_classes * [0]
                    }
                    sample_accum.append(no_annotation)
                    if len(sample_accum) == reduce:
                        data.append(sample_accum)
                        sample_accum = []
                if sample_accum != []:
                    data.append(sample_accum)
                self.data.extend(data)
            random.shuffle(self.data)
            self.logger.info(f'len self.data[base_img_dir={base_img_dir}] = {len(self.data)}')

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

        self.logger.info(f'len dataset = {len(self.data)}')
        self.logger.info(f'{self.__class__.__qualname__} initiated')

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, str]:

        try:
            sample = np.random.choice(self.data[idx])
            img_path = Path(sample["path"])
            img = img_read_PIL_RGB(img_path)
        except Exception as excpt:
            msg = f'Some error occurred with image: {self.data[idx]}'
            self.logger.error(msg)
            self.logger.error(excpt)
            raise excpt
        img = np.uint8(img)

        masks = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        img, _, _ = self.transforms(img, masks)

        # We accept -1 as not filled by annotators.
        tgt_clf = -torch.ones(self.n_classes).long()
        values_sub_names = sample["annotations"]
        for idx, value_sub_name in enumerate(values_sub_names):
            tgt_clf[idx] = value_sub_name

        return img, tgt_clf, img_path.stem

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor, List[str]]:
        s_image, s_tgt_clf, s_fn = [], [], []
        for image, tgt_clf, tgt_reg, fn in batch:
            s_image.append(image)
            s_tgt_clf.append(tgt_clf)
            s_fn.append(fn)
        s_image = torch.stack(s_image, dim=0)
        s_tgt_clf = torch.stack(s_tgt_clf, dim=0)
        return s_image, s_tgt_clf, s_fn
