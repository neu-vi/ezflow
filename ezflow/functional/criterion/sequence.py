import torch
import torch.nn as nn

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class SequenceLoss(nn.Module):
    @configurable
    def __init__(self, gamma=0.8, max_flow=400):
        super(SequenceLoss, self).__init__()

        self.gamma = gamma
        self.max_flow = max_flow

    @classmethod
    def from_config(cls, cfg):
        return {"gamma": cfg.GAMMA, "max_flow": cfg.MAX_FLOW}

    def forward(self, pred, label):

        n_preds = len(pred)
        flow_loss = 0.0

        mag = torch.sqrt(torch.sum(label ** 2, dim=1))

        # Temp Fix
        valid = (label[:, 0].abs() < 1000) & (label[:, 1].abs() < 1000)
        valid = valid.float()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_preds):

            i_weight = self.gamma ** (n_preds - i - 1)
            i_loss = torch.abs(pred[i] - label)
            flow_loss += i_weight * torch.mean((valid[:, None] * i_loss))

        return flow_loss
