import inspect
import logging
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

import segmentation_models_pytorch as smp


class L2:
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    def __call__(self, pred_clf: Tensor, out_reg: Tensor, tgt_clf: Tensor, tgt_reg: Tensor) -> Tuple[np.ndarray, int]:
        out_reg = out_reg.view(-1)
        tgt_reg = tgt_reg.view(-1)

        loss_reg = torch.pow(out_reg - tgt_reg, 2)
        mask_annotated = tgt_reg >= 0
        loss_reg = loss_reg[mask_annotated]
        loss_reg = torch.sqrt(loss_reg.mean())

        return loss_reg.cpu().numpy(), mask_annotated.sum().item()


class BinaryAccuracy:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_clf: Tensor, out_reg: Tensor, tgt_clf: Tensor, tgt_reg: Tensor) -> Tuple[np.ndarray, int]:
        bs, nc = pred_clf.shape[:2]
        pred_clf = pred_clf.view(bs, nc)
        tgt_clf = tgt_clf.view(bs, nc)
        mask_annotated = tgt_clf >= 0

        pred_clf = pred_clf > self.threshold
        acc = (pred_clf == tgt_clf).to(torch.float).sum(0) / mask_annotated.sum(0)

        return acc.cpu().numpy(), mask_annotated.sum().item()


class BinaryTpr:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_clf: Tensor, out_reg: Tensor, tgt_clf: Tensor, tgt_reg: Tensor) -> Tuple[np.ndarray, int]:
        bs, nc = pred_clf.shape[:2]
        pred_clf = pred_clf.view(bs, nc)
        tgt_clf = tgt_clf.view(bs, nc)

        pred_clf = pred_clf > self.threshold
        pos = tgt_clf == 1
        tp = (pred_clf[pos] == 1).to(torch.float).sum(0)
        fn = (pred_clf[pos] == 0).to(torch.float).sum(0)

        n_pos = pos.to(torch.float).sum().item()
        tpr = (tp / (tp + fn)).cpu().numpy()
        if n_pos > 0:
            return tpr, n_pos
        else:
            tpr[...] = 0
            return tpr, 0


class BinaryTnr:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_clf: Tensor, out_reg: Tensor, tgt_clf: Tensor, tgt_reg: Tensor) -> Tuple[np.ndarray, int]:
        bs, nc = pred_clf.shape[:2]
        pred_clf = pred_clf.view(bs, nc)
        tgt_clf = tgt_clf.view(bs, nc)

        pred_clf = pred_clf > self.threshold
        neg = tgt_clf != 1
        tn = (pred_clf[neg] == 0).to(torch.float).sum(0)
        fp = (pred_clf[neg] == 1).to(torch.float).sum(0)

        n_neg = neg.to(torch.float).sum().item()
        tnr = (tn / (tn + fp)).cpu().numpy()
        if n_neg > 0:
            return tnr, n_neg
        else:
            tnr[...] = 0
            return tnr, 0


class SMPperImgIOU:
    def __init__(self, threshold):
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_mask: torch.FloatTensor, targets_mask: Tensor) -> Tuple[Tensor, int]:
        with torch.no_grad():
            assert targets_mask.ndim == 4
            assert pred_mask.ndim == 4
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask, targets_mask, threshold=self.threshold, mode="binary")
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            return per_image_iou, pred_mask.shape[0]


class SMPperImgSpecificity:
    def __init__(self, threshold):
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_mask: torch.FloatTensor, targets_mask: Tensor) -> Tuple[Tensor, int]:
        with torch.no_grad():
            is_neg = targets_mask.sum([1, 2, 3]) == 0
            count = is_neg.sum().item()
            if count > 0:
                pred_mask = pred_mask[is_neg]
                assert pred_mask.ndim == 4
                targets_mask = targets_mask[is_neg]
                assert targets_mask.ndim == 4
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mask, targets_mask, threshold=self.threshold, mode="binary")
                per_image_specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
                return per_image_specificity, count
            else:
                return torch.tensor(0).to(pred_mask.dtype), 0


class SMPperImgSensitivity:
    def __init__(self, threshold):
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 <= threshold < 1:
            self.logger.error("threshold must be in the range [0, 1], however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, pred_mask: torch.FloatTensor, targets_mask: Tensor) -> Tuple[Tensor, int]:
        with torch.no_grad():
            is_pos = targets_mask.sum([1, 2, 3]) > 0
            count = is_pos.sum().item()
            if count > 0:
                pred_mask = pred_mask[is_pos]
                assert pred_mask.ndim == 4
                targets_mask = targets_mask[is_pos]
                assert targets_mask.ndim == 4
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mask, targets_mask, threshold=self.threshold, mode="binary")
                per_image_sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
                return per_image_sensitivity, count
            else:
                return torch.tensor(0).to(pred_mask.dtype), 0
