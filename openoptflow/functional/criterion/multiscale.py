import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class MultiScaleLoss(nn.Module):
    @configurable
    def __init__(
        self,
        norm="l1",
        weights=(1, 0.5, 0.25),
        extra_mask=None,
        use_valid_range=True,
        valid_range=None,
    ):
        super(MultiScaleLoss, self).__init__()

        self.norm = norm.lower()
        assert self.norm in ("l1", "l2", "robust"), "Norm must be one of L1, L2, Robust"

        self.weights = weights
        self.extra_mask = extra_mask
        self.use_valid_range = use_valid_range
        self.valid_range = valid_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "norm": cfg.NORM,
            "weights": cfg.WEIGHTS,
            "extra_mask": cfg.EXTRA_MASK,
            "use_valid_range": cfg.USE_VALID_RANGE,
            "valid_range": cfg.VALID_RANGE,
        }

    def forward(self, pred, label):

        loss = 0
        h, w = label.size()[-2:]

        if (
            (type(pred) is not tuple)
            and (type(pred) is not list)
            and (type(pred) is not set)
        ):
            pred = {pred}

        for i, level_pred in enumerate(pred):

            real_flow = F.interpolate(
                level_pred, (h, w), mode="bilinear", align_corners=True
            )
            real_flow[:, 0, :, :] = real_flow[:, 0, :, :] * (w / level_pred.shape[3])
            real_flow[:, 1, :, :] = real_flow[:, 1, :, :] * (h / level_pred.shape[2])

            if self.norm == "l2":
                loss_value = torch.norm(real_flow - label, p=2, dim=1)

            elif self.norm == "robust":
                loss_value = (real_flow - label).abs().sum(dim=1) + 1e-8
                loss_value = loss_value ** 0.4

            elif self.norm == "l1":
                loss_value = (real_flow - label).abs().sum(dim=1)

            if self.use_valid_range and self.valid_range is not None:

                with torch.no_grad():
                    mask = (label[:, 0, :, :].abs() <= self.valid_range[i][1]) & (
                        label[:, 1, :, :].abs() <= self.valid_range[i][0]
                    )
            else:
                with torch.no_grad():
                    mask = torch.ones(label[:, 0, :, :].shape).type_as(label)

            loss_value = loss_value * mask.float()

            if self.extra_mask is not None:
                val = self.extra_mask > 0
                loss_value = loss_value[val]
                level_loss = loss_value.mean() * self.weights[i]

            else:
                level_loss = loss_value.mean() * self.weights[i]

            loss += level_loss

        loss = loss / len(pred)

        return loss
