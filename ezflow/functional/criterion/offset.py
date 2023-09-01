import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY
from .sequence import SequenceLoss


@FUNCTIONAL_REGISTRY.register()
class OffsetCrossEntropyLoss(nn.Module):
    @configurable
    def __init__(
        self,
        stride=8,
        offset_loss_weight=[0, 1],
        weight_anneal_fn="CosineAnnealer",
        min_weight=0,
        max_iter=1,
        **kwargs,
    ):
        super(OffsetCrossEntropyLoss, self).__init__()
        assert (
            weight_anneal_fn in FUNCTIONAL_REGISTRY
        ), f"{weight_anneal_fn} not found. Available weight annelers {FUNCTIONAL_REGISTRY.get_list()}"

        self.stride = stride

        weight_anneal_fn = FUNCTIONAL_REGISTRY.get(weight_anneal_fn)
        self.weight_annealers = [
            weight_anneal_fn(init_weight=wt, min_weight=min_weight, max_iter=max_iter, **kwargs) for wt in offset_loss_weight
        ]

    @classmethod
    def from_config(cls, cfg):
        return {
            "stride": cfg.STRIDE,
            "weight_anneal_fn": cfg.WEIGHT_ANNEAL_FN,
            "offset_loss_weight": cfg.OFFSET_LOSS_WEIGHT,
            "min_weight": cfg.MIN_WEIGHT,
            "max_iter": cfg.MAX_ITER
        }

    def __compute_loss(self, flow_logits, offset_labs, valid):
        # exlude invalid pixels and extremely large diplacements()
        valid = torch.squeeze(valid, dim=1)
        valid = valid[:, :: self.stride, :: self.stride]
        valid = valid >= 0.5

        logprobs = F.log_softmax(flow_logits, dim=1)
        loss = -(offset_labs * logprobs).sum(dim=1)
        loss = (valid[:, None] * loss).mean()
        return loss

    def forward(self, flow_logits_list, offset_labs, valid, current_iter, **kwargs):
        logit_loss = 0.0
        for i, flow_logits in enumerate(flow_logits_list):
            loss = self.__compute_loss(flow_logits, offset_labs, valid)
            logit_loss += (self.weight_annealers[i](current_iter)) * loss

        return logit_loss


@FUNCTIONAL_REGISTRY.register()
class FlowOffsetLoss(nn.Module):
    @configurable
    def __init__(
        self,
        gamma=0.25,
        max_flow=500,
        stride=8,
        weight_anneal_fn="CosineAnnealer",
        offset_loss_weight=[0, 1],
        min_weight=0,
        max_iter=1,
        **kwargs,
    ):
        super(FlowOffsetLoss, self).__init__()

        self.l1_loss = SequenceLoss(gamma=gamma, max_flow=max_flow)
        self.cross_entropy_loss = OffsetCrossEntropyLoss(
            offset_loss_weight=offset_loss_weight,
            weight_anneal_fn=weight_anneal_fn,
            stride=stride,
            min_weight=min_weight,
            max_iter=max_iter,
            **kwargs,
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "gamma": cfg.GAMMA,
            "max_flow": cfg.MAX_FLOW,
            "stride": cfg.STRIDE,
            "weight_anneal_fn": cfg.WEIGHT_ANNEAL_FN,
            "offset_loss_weight": cfg.OFFSET_LOSS_WEIGHT,
            "min_weight": cfg.MIN_WEIGHT,
            "max_iter": cfg.MAX_ITER
        }

    def forward(
        self,
        flow_preds,
        flow_logits,
        flow_gt,
        valid,
        offset_labs,
        current_iter,
        **kwargs,
    ):

        flow_loss = self.l1_loss(flow_preds, flow_gt, valid)
        logit_loss = self.cross_entropy_loss(
            flow_logits, offset_labs, valid, current_iter, **kwargs
        )
        loss = flow_loss + logit_loss
        return loss
