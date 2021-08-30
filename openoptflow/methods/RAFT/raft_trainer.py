import torch
import torch.nn as nn

from ...base_trainer import BaseTrainer


class SequenceLoss(nn.Module):
    def __init__(self, gamma=0.8, max_flow=400):
        super(SequenceLoss, self).__init__()

        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, pred, label):

        n_preds = len(pred)
        flow_loss = 0.0

        mag = torch.sqrt(torch.sum(label ** 2, dim=1))

        valid = (torch.abs(label[0]) < 1000) and (torch.abs(label[1]) < 1000)
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_preds):

            i_weight = self.gamma ** (n_preds - i - 1)
            i_loss = torch.abs(pred[i] - label)
            flow_loss += i_weight * torch.mean((valid[:, None] * i_loss))

        return flow_loss


class RAFTTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(RAFTTrainer, self).__init__()

        self.loss_fn = SequenceLoss(**kwargs)

    def _calculate_loss(self, pred, label):

        return self.loss_fn(pred, label)
