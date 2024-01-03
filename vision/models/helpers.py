#!/usr/bin/env python
import gc
import inspect
import logging
import math
import os
from typing import Union

import torch
from path import Path
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module

from vision.optimizer import RETURN_optimizer_builder
from vision.scheduler import RETURN_scheduler_builder
from vision.utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel


def checkpoint_dictionary(epoch: int, loss: float, metric: float, patient: int, model: Module,
                          lr_scheduler: RETURN_scheduler_builder = None, scaler: GradScaler = None,
                          optimizer: RETURN_optimizer_builder = None) -> dict:
    checkpoint = {'epoch': epoch, 'patient': patient, 'loss': loss, 'metric': metric, 'model': model.state_dict(),
                  'scaler': scaler.state_dict() if scaler is not None else None,
                  'optimizer': optimizer.state_dict() if optimizer is not None else None,
                  'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None}
    return checkpoint


def save_checkpoint(save_model_file: str, epoch, patient, model: Module, loss: float, metric: float,
                    lr_scheduler: RETURN_scheduler_builder = None, optimizer: RETURN_optimizer_builder = None,
                    scaler: GradScaler = None) -> [int, int, dict]:
    checkpoint = checkpoint_dictionary(epoch=epoch, loss=loss, metric=metric, patient=patient, model=model,
                                       lr_scheduler=lr_scheduler, optimizer=optimizer, scaler=scaler)
    save_model_file = Path(save_model_file)
    save_model_file.abspath().parent.makedirs_p()
    torch.save(checkpoint, save_model_file)


def load_checkpoint(model: Module, checkpoint_path: Union[str, Path], lr_scheduler: RETURN_scheduler_builder = None,
                    optimizer: RETURN_optimizer_builder = None, scaler: GradScaler = None,
                    device: torch.device = torch.device('cpu'), load_optimizer: bool = True,
                    load_scheduler: bool = True, strict: bool = True, ) -> [int, int, dict, dict]:
    """
    Checkpoint loader.

    Args:
        model: The loaded weights will be saved to this model.
        checkpoint_path: Path to the checkpoint file.
        lr_scheduler: The scheduler information will be saved to this learning rate scheduler.
        optimizer: The optimizer data will be saved to this optimizer.
        scaler: The scaler data will be saved to this scaler.
        device: Device to load the information to.
        load_optimizer: Whether to load the optimizer.
        load_scheduler: Whether to load the scheduler.
        strict: Whether to load the model, optimizer, and scaler in strict mode.

    Returns:
        epoch, patient, loss, metric
        Last epoch, patient, and best loss and best metric.
    """
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    checkpoint_path = os.path.expanduser(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        epoch = checkpoint.get('epoch', 0)
        patient = checkpoint.get('patient', 0)
        loss = checkpoint.get('loss', None)
        metric = checkpoint.get('metric', None)
        strict = True if strict is None else strict

        try:
            state_dict = checkpoint['model']
            if not is_model_dataparallel(model):
                state_dict = rmv_module_dataparallel(state_dict)
            model.load_state_dict(state_dict, strict=strict)
            logger.info(f'Loaded checkpoint {checkpoint_path} at epoch {epoch}')
        except RuntimeError as excpt:
            if strict:
                msg = f'It was not possible to load the checkpoint {checkpoint_path}\n'
                msg += excpt
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(f'It was not possible to load checkpoint {checkpoint_path}')

        if optimizer is not None:
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None and load_optimizer:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except RuntimeError as excpt:
                    if strict:
                        raise excpt
                    logger.warning(" ***** optimizer will start from scratch ***** ")
            else:
                logger.warning(" ***** optimizer will start from scratch ***** ")

        if lr_scheduler is not None:
            if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None and load_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                logger.warning(" ***** lr_scheduler will start from scratch ***** ")

        if scaler is not None:
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                try:
                    scaler.load_state_dict(checkpoint['scaler'])
                except RuntimeError as excpt:
                    if strict:
                        raise excpt
                    logger.warning(" ***** scaler will start from scratch ***** ")
            else:
                logger.warning(" ***** scaler will start from scratch ***** ")

        del checkpoint
        gc.collect()

        return epoch, patient, loss, metric

    else:
        logger.warning(f'File config["load_model_file"] = {checkpoint_path} does not exist.')
        logger.warning('  ************************************************   ')
        logger.warning('  ****** Start training from scratch         *****   ')
        logger.warning('  ************************************************   ')
        return 0, 0, math.inf, None
