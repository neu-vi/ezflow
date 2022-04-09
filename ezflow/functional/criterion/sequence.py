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
    def __init__(self, gamma=0.8, max_flow=400):
        super(SequenceLoss, self).__init__()

        self.gamma = gamma
        self.max_flow = max_flow

    @classmethod
    def from_config(cls, cfg):
        return {"gamma": cfg.GAMMA, "max_flow": cfg.MAX_FLOW}

    def forward(self, pred, label):
        assert (
            label.shape[1] == 3
        ), "Incorrect channel dimension. Set append valid mask to True in DataloaderCreator to append the valid data mask in the target label."

        n_preds = len(pred)
        flow_loss = 0.0

        flow, valid = label[:, :2, :, :], label[:, 2:, :, :]
        valid = torch.squeeze(valid, dim=1)

        mag = torch.sqrt(torch.sum(flow ** 2, dim=1))
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_preds):

            i_weight = self.gamma ** (n_preds - i - 1)
            i_loss = torch.abs(pred[i] - flow)
            flow_loss += i_weight * torch.mean((valid[:, None] * i_loss))

        return flow_loss
