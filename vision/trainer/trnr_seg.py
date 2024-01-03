#!/usr/bin/env python
import gc
import inspect
import json
import logging
import time
from typing import Dict, Optional, Callable

import numpy as np
import torch
from path import Path
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vision.losses_metrics import AverageMeter
from vision.models.helpers import save_checkpoint
from vision.optimizer import RETURN_optimizer_builder
from vision.optimizer.clip import Clipper
from vision.scheduler import RETURN_scheduler_builder
from vision.utils.torch.dataparallel import reduce_average_meter


def trainer(epoch: int,
            data_loader: DataLoader,
            model: Module,
            optimizer: RETURN_optimizer_builder,
            lr_scheduler: RETURN_scheduler_builder,
            step_scheduler_at_save: bool,
            loss_funct: Module,
            metric_functs: Optional[Dict[str, Callable]],
            scaler: GradScaler = None,
            writer: SummaryWriter = None,
            fn_resume: str = None,
            stage: str = "train",
            clipper: Clipper = None,
            grad_accum: int = 1,
            this_gpu: int = 0,
            rank: int = 0,
            n_log_interval: int = 100,
            n_save_inter_epoch: int = 100,
            save_tmp_model_fn: str = None,
            non_blocking: bool = True,
            distributed_data_parallel: bool = False,
            ) -> (float, Dict[str, float], Module):
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    # Select whether trainable or not.
    if stage == "train":
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # Meters.
    loss_meter = AverageMeter()
    classes = data_loader.dataset.classes
    metric_meters = {metric_name: dict() for metric_name in metric_functs.keys()}
    for metric_name in metric_functs.keys():
        metric_meters[metric_name].update({cls_name: AverageMeter(accept_zero_samples=True) for cls_name in classes})
    fn_resume = Path(fn_resume)
    resume_it = dict()

    # Loop over mini-batches.
    logger.info(f"{stage} Ep={epoch}")
    it = None
    start_time = time.time()
    for it, batch in enumerate(data_loader):
        img, gt_mask, consider, fn = batch
        bs = img.shape[0]
        img = img.detach().requires_grad_(requires_grad=False).cuda(this_gpu, non_blocking=non_blocking)
        gt_mask = gt_mask.squeeze(1).long().detach().requires_grad_(requires_grad=False).cuda(
            this_gpu, non_blocking=non_blocking)
        consider = consider.long().detach().requires_grad_(requires_grad=False).cuda(this_gpu, non_blocking=non_blocking)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float32 if scaler is None else torch.float16):
                logits_mask = model(img)
                loss = loss_funct(logits_mask, gt_mask, consider)
        else:
            logits_mask = model(img)
            loss = loss_funct(logits_mask, gt_mask, consider)

        if torch.isnan(loss):
            msg = f"NaN loss found at it={it}"
            logger.error(msg)
            raise ValueError(msg)

        if stage == "train":
            loss = (1. / grad_accum) * loss
            if scaler is not None:
                scaler.scale(loss).backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        scaler.unscale_(optimizer)
                        clipper.apply_to_grad(model)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        clipper.apply_to_grad(model)
                    optimizer.step()
                    optimizer.zero_grad()

        loss_meter.update(grad_accum * loss.item(), bs)

        pred_mask = logits_mask.sigmoid()
        with torch.no_grad():
            for metric_name, metric_funct in metric_functs.items():
                for ib in range(bs):
                    for cls_name, metric_meter in metric_meters[metric_name].items():
                        icls = classes.index(cls_name)
                        if consider[ib, icls] > 0:
                            metric, cnt = metric_funct(
                                pred_mask[ib, icls, :, :].unsqueeze(0).unsqueeze(0),
                                gt_mask[ib, icls, :, :].unsqueeze(0).unsqueeze(0)
                            )
                            metric = metric.item()
                            assert cnt in [0, 1]
                            metric_meter.update(metric, cnt)

        # Intermediate results.
        if (it + 1) % n_log_interval == 0:
            time_elapse = 1000 * (time.time() - start_time)
            logger.info(f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f} | "
                        f"{time_elapse:.5}ms cuda:{this_gpu}")
            logger.info(45 * '-')
            if rank == 0:
                for metric_name, _ in metric_functs.items():
                    for cls_name, metric_meter in metric_meters[metric_name].items():
                        logger.info(
                            f"{metric_name[9:]} | {cls_name[:14]: <15} | MA {metric_meter.val: .3f} | MT {metric_meter.avg: .3f} | CNT {metric_meter.count}")
                    logger.info(45 * '-')
            start_time = time.time()

        if (it + 1) % n_save_inter_epoch == 0:
            if step_scheduler_at_save:
                lr_scheduler.step()
                logger.info(' ************************************************** ')
                logger.info(f' ***** lr = {optimizer.param_groups[0]["lr"]} ***** ')
                logger.info(' ************************************************** ')
            if rank == 0:
                logger.info(f'Saving temporal model {save_tmp_model_fn}')
                save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                                lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)
                save_checkpoint(save_tmp_model_fn[:-5] + f"_it={it}.ckpt", model=model, optimizer=optimizer,
                                scaler=scaler, lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

            gc.collect()
            torch.cuda.memory_reserved()

    # Final results.
    if distributed_data_parallel and torch.cuda.device_count() > 1:
        for metric_name, _ in metric_functs.items():
            for cls_name, metric_meter in metric_meters[metric_name].items():
                reduce_average_meter(metric_meter, this_gpu)
        reduce_average_meter(loss_meter, this_gpu)

    logger.info(f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f}")
    logger.info(45 * '-')
    if rank == 0:
        for metric_name, _ in metric_functs.items():
            for cls_name, metric_meter in metric_meters[metric_name].items():
                resume_it.setdefault(metric_name, dict()).update({cls_name: metric_meter.avg})
                logger.info(
                    f"{metric_name[9:]} | {cls_name[:14]: <15} | MA {metric_meter.val: .3f} | MT {metric_meter.avg: .3f} | CNT {metric_meter.count}")
            logger.info(45 * '-')

    if writer is not None:
        step = it + epoch * len(data_loader)
        writer.add_scalar(f"Loss/{stage}", loss_meter.avg, global_step=step)
        for metric_name, _ in metric_functs.items():
            for cls_name, metric_meter in metric_meters[metric_name].items():
                writer.add_scalar(f"Metric/{metric_name}/{cls_name}", metric_meter.avg, global_step=step)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step=step)

    logger.info(f'Saving temporal model {save_tmp_model_fn}')
    save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                    lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

    resume = json.load(open(fn_resume)) if fn_resume.exists() else dict()
    resume.setdefault(stage, []).append(resume_it)
    json.dump(resume, open(fn_resume, 'w'), indent=4)

    metric_meter_avg = np.mean([_.avg for _ in metric_meters["SMPperImgIOU"].values()])
    return loss_meter.avg, metric_meter_avg, model
