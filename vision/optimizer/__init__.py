import inspect
import logging
from typing import Union

import torch
from easydict import EasyDict
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW

from submodules.SimulatedAnnealing.optim.sa import GaussianSampler
from submodules.SimulatedAnnealing.optim.sa import SimulatedAnnealing
from torch_pso import ParticleSwarmOptimizer

RETURN_optimizer_builder = Union[SGD, Adam, AdamW, ParticleSwarmOptimizer, SimulatedAnnealing]


def get_optimizer(model: Module, cfg: dict = None) -> RETURN_optimizer_builder:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    cfg = EasyDict(cfg)
    if not cfg:
        cfg.name = 'SGD'
        cfg.args = EasyDict(lr=0.1)
        logger.warning(f'Optimizer is not defined. Using default optimizer: {cfg.name} with lr: {cfg.args.lr}')

    if cfg.name == 'SGD':
        return SGD(model.parameters(), **cfg.args)
    elif cfg.name == 'Adam':
        return Adam(model.parameters(), **cfg.args)
    elif cfg.name == 'AdamW':
        return AdamW(model.parameters(), **cfg.args)
    elif cfg.name == 'SimulatedAnnealing':
        assert torch.cuda.device_count() == 1
        sampler = GaussianSampler(mu=0, sigma=1, cuda=True)
        return SimulatedAnnealing(model.parameters(), sampler=sampler)
    elif cfg.name == 'ParticleSwarmOptimizer':
        assert torch.cuda.device_count() == 1
        return ParticleSwarmOptimizer(model.parameters(), **cfg.args)
    else:
        msg = f'Not implemented optimizer: name = {cfg.name}'
        logger.error(msg)
        raise NotImplementedError(msg)
