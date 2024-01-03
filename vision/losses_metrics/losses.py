from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import BCEWithLogitsLoss

from submodules.segmentation_models_pytorch.segmentation_models_pytorch.losses._functional import \
    soft_dice_score, focal_loss_with_logits
from submodules.segmentation_models_pytorch.segmentation_models_pytorch.losses.constants import \
    BINARY_MODE

_DEBUG = False


class L1(Module):
    def __init__(self, err_min=None):
        super().__init__()
        self.err_min = err_min

    def __call__(self, out_reg: Tensor, tgt_reg: Tensor) -> Tensor:
        out_reg = out_reg.view(-1)
        tgt_reg = tgt_reg.view(-1)

        loss_reg = torch.abs(out_reg - tgt_reg)
        if self.err_min is not None:
            loss_reg[loss_reg < self.err_min] *= 0
        mask_annotated = tgt_reg >= 0
        loss_reg = loss_reg[mask_annotated]
        loss_reg = loss_reg.mean()

        return loss_reg


class L2(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, out_reg: Tensor, tgt_reg: Tensor) -> Tensor:
        out_reg = out_reg.view(-1)
        tgt_reg = tgt_reg.view(-1)

        loss_reg = torch.pow(out_reg - tgt_reg, 2)
        mask_annotated = tgt_reg >= 0
        loss_reg = loss_reg[mask_annotated]
        loss_reg = loss_reg.mean()

        return loss_reg


class BCELogitsLoss(Module):
    def __init__(self, n_classes=None, pos_weight=None) -> None:
        super().__init__()
        if pos_weight is not None:
            pos_weight = np.array(pos_weight).reshape(n_classes)
            pos_weight = torch.FloatTensor(pos_weight).cuda()
        self.loss = BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(outputs.dtype)
        mask_annotated = targets >= 0
        loss = mask_annotated * self.loss(outputs, targets)
        loss = loss.mean()
        return loss


class DicePlusFocalLoss(Module):
    def __init__(self,
                 mode, from_logits,
                 mean_pos_focal, mean_neg_focal, mean_pos_dice, mean_neg_dice,
                 weight_cls, weight_f_d
                 ):
        super().__init__()
        assert from_logits  # focal loss is only implemented with logits
        assert mode == "binary"

        weight_f_d = np.array(weight_f_d)
        self.weight_f_d = weight_f_d / weight_f_d.sum()

        weight_cls = np.array(weight_cls)
        self.weight_cls = weight_cls / weight_cls.sum()

        # Focal
        mean_pos_focal = np.array(mean_pos_focal)
        mean_neg_focal = np.array(mean_neg_focal)
        self.weights_pos_focal = mean_pos_focal / (mean_pos_focal + mean_neg_focal)

        if self.weight_f_d[0] > 0:
            self.loss_focal = [
                BCEWithLogitsLoss(pos_weight=torch.FloatTensor([self.weights_pos_focal[icls]]).cuda(), reduction="none")
                for icls in range(len(self.weights_pos_focal))
            ]

        # Dice
        mean_pos_dice = np.array(mean_pos_dice)
        mean_neg_dice = np.array(mean_neg_dice)
        self.weights_pos_dice = mean_pos_dice / (mean_pos_dice + mean_neg_dice)
        self.weights_neg_dice = mean_neg_dice / (mean_pos_dice + mean_neg_dice)

        if self.weight_f_d[1] > 0:
            self.loss_dice = DiceLoss(mode=mode, from_logits=from_logits)

    def print(self, msg=""):
        if _DEBUG:
            print(msg)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, consider: torch.Tensor) -> torch.Tensor:
        bs = y_pred.shape[0]
        nc = y_pred.shape[1]
        device = y_pred.device
        dtype = y_pred.dtype

        loss = torch.tensor(0.).to(dtype).to(device)
        self.print()
        for icls in range(nc):
            pred = y_pred[:, icls, :, :].unsqueeze(1).contiguous()
            gt = y_true[:, icls, :, :].unsqueeze(1).contiguous()
            self.print(f"icls = {icls}")
            self.print(f"self.weights_pos_dice = {[_.item() for _ in list(self.weights_pos_dice)]}")
            self.print(f"self.weights_neg_dice = {[_.item() for _ in list(self.weights_neg_dice)]}")

            loss_dice = 0
            if self.weight_f_d[1] > 0:
                is_pos = gt.sum([1, 2, 3]) > 0
                self.print(f"is_pos = {[_.item() for _ in list(is_pos)]}")
                weight_dice = torch.ones(bs, requires_grad=False).to(dtype).to(device)
                weight_dice[is_pos] = self.weights_pos_dice[icls]
                weight_dice[~is_pos] = self.weights_neg_dice[icls]
                self.print(f"weight_dice.shape weight_dice.sum() = {weight_dice.shape} {weight_dice.sum()}")
                loss_dice = self.loss_dice(pred, gt).reshape(bs)
                self.print(f"loss_dice = {[_.item() for _ in list(loss_dice)]}")
                loss_dice = (weight_dice * loss_dice).reshape(bs)
                self.print(f"loss_dice.shape = {loss_dice.shape}")
                self.print(f"consider.shape consider.sum() = {consider.shape} {consider.sum()}")
                loss_dice = (consider[:, icls] * loss_dice).reshape(bs)
                self.print(f"loss_dice.shape = {loss_dice.shape}")
                loss_dice = loss_dice.mean()
                self.print(f"loss_dice = {loss_dice}")
                self.print(f"loss_dice.shape = {loss_dice.shape}")

            loss_focal = 0
            if self.weight_f_d[0] > 0:
                gt = gt.to(pred.dtype)
                loss_focal = self.loss_focal[icls](pred, gt).mean([1, 2, 3]).reshape(bs)
                self.print(f"loss_focal = {[_.item() for _ in list(loss_focal)]}")
                self.print(f"loss_focal.shape = {loss_focal.shape}")
                self.print(f"consider.shape consider.sum() = {consider.shape} {consider.sum()}")
                loss_focal = (consider[:, icls] * loss_focal).reshape(bs)
                self.print(f"loss_focal.shape = {loss_focal.shape}")
                loss_focal = loss_focal.mean()
                self.print(f"loss_focal = {loss_focal}")
                self.print(f"loss_focal.shape = {loss_focal.shape}")

            loss_focal *= self.weight_f_d[0]
            loss_dice *= self.weight_f_d[1]
            self.print(f"loss_focal = {loss_focal}")
            self.print(f"loss_dice = {loss_dice}")
            loss += self.weight_cls[icls] * (loss_focal + loss_dice)
        self.print(f"loss = {loss}")
        # exit()

        return loss


class DiceLoss(Module):
    def __init__(
            self,
            mode: str,
            from_logits: bool = True,
            smooth: float = 0.0,
            eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode

        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            # y_pred = F.logsigmoid(y_pred).exp()
            y_pred = functional.sigmoid(y_pred)

        bs = y_true.size(0)
        assert y_pred.size(1) == 1
        assert y_pred.size(1) == 1
        dims = (2,)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return loss

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)


class FocalLoss(Module):
    def __init__(
            self,
            mode: str,
            alpha: Optional[float] = None,
            gamma: Optional[float] = 2.0,
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE}
        super().__init__()

        self.mode = mode
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction="none",
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.focal_loss_fn(y_pred, y_true)
        return loss
