import math

from ..config import configurable
from .registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class CosineAnnealer(object):
    @configurable
    def __init__(self, init_weight, min_weight, max_iter):
        self.init_weight = init_weight
        self.min_weight = min_weight
        self.max_iter = max_iter

    @classmethod
    def from_config(cls, cfg):
        return {
            "init_weight": cfg.INIT_WEIGHT,
            "min_weight": cfg.MAX_weight,
            "max_iter": cfg.MAX_ITER,
        }

    def __call__(self, cur_iter):
        wt = (
            self.min_weight
            + (self.init_weight - self.min_weight)
            * (1 + math.cos(math.pi * cur_iter / self.max_iter))
            / 2
        )
        return wt


@FUNCTIONAL_REGISTRY.register()
class PolyAnnealer(object):
    @configurable
    def __init__(self, init_weight, min_weight, max_iter, power):
        self.init_weight = init_weight
        self.min_weight = min_weight
        self.max_iter = max_iter
        self.power = power

    @classmethod
    def from_config(cls, cfg):
        return {
            "init_weight": cfg.INIT_WEIGHT,
            "min_weight": cfg.MAX_weight,
            "max_iter": cfg.MAX_ITER,
            "power": cfg.POWER,
        }

    def __call__(self, cur_iter):
        wt = (self.init_weight - self.min_weight) * (
            (1 - cur_iter / self.max_iter) ** (self.power)
        ) + self.min_weight
        return wt
