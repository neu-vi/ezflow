import torch
import torch.nn as nn

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class SequenceLoss(nn.Module):
    """
    Sequence loss for optical flow estimation.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    gamma : float
        Weight for the loss
    max_flow : float
        Maximum flow magnitude
    """

    @configurable
    def __init__(self, gamma=0.8, max_flow=400, **kwargs):
        super(SequenceLoss, self).__init__()

        self.gamma = gamma
        self.max_flow = max_flow

    @classmethod
    def from_config(cls, cfg):
        return {"gamma": cfg.GAMMA, "max_flow": cfg.MAX_FLOW}

    def forward(self, flow_preds, flow_gt, valid, **kwargs):

        n_preds = len(flow_preds)
        flow_loss = 0.0
        valid = torch.squeeze(valid, dim=1)

        mag = torch.sqrt(torch.sum(flow_gt**2, dim=1))
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_preds):

            i_weight = self.gamma ** (n_preds - i - 1)
            i_loss = torch.abs(flow_preds[i] - flow_gt)
            flow_loss += i_weight * torch.mean((valid[:, None] * i_loss))

        return flow_loss
