import inspect
import logging
import math
from typing import Callable

from torch.nn import Module

from vision.losses_metrics import losses, metrics


def get_loss(name: str, kwargs=None, rank=0) -> Module:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if kwargs is None:
        kwargs = dict()

    losses_available = [
        losses.L1,
        losses.L2,
        losses.DiceLoss,
        losses.FocalLoss,
        losses.DicePlusFocalLoss,
        losses.BCELogitsLoss,
    ]

    for loss_available in losses_available:
        if name == loss_available.__name__:
            return loss_available(**kwargs)

    msg = f'Wrong loss name {name}.'
    logger.error(msg)
    raise ValueError(msg)


def get_metric(name: str, kwargs=None) -> Callable:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if kwargs is None:
        kwargs = dict()

    metrics_available = [
        metrics.SMPperImgIOU,
        metrics.SMPperImgSensitivity,
        metrics.SMPperImgSpecificity,
        metrics.BinaryAccuracy,
        metrics.BinaryTpr,
        metrics.BinaryTnr,
        metrics.L2,
    ]

    for metrics_availabl in metrics_available:
        if name == metrics_availabl.__name__:
            return metrics_availabl(**kwargs)

    msg = f'Wrong metric name {name}.'
    logger.error(msg)
    raise ValueError(msg)


class AverageMeter:
    """Stores and computes  the average and current value"""

    def __init__(self, accept_zero_samples: bool = False) -> None:
        self.val: float = math.inf
        self.avg: float = 0.
        self.sum: float = 0.
        self.count: int = 0
        self.accept_zero_samples = accept_zero_samples

    def reset(self) -> None:
        self.val: float = math.inf
        self.avg: float = 0.
        self.sum: float = 0.
        self.count: int = 0

    def update(self, value: float, batch_size: int = 1) -> None:
        logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        if batch_size == 0 and not self.accept_zero_samples:
            msg = 'Zero values passed are not allowed to compute a mean value.'
            logger.error(msg)
            raise ValueError(msg)

        if batch_size != 0:
            self.val = value
            self.sum += value * batch_size
            self.count += batch_size
            self.avg = self.sum / (self.count + 1.e-6)
