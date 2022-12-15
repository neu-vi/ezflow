import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for optical flow estimation.
    Used in **DICL** (https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html)

    Parameters
    ----------
    norm : str, default: "l1"
        The norm to use for the loss. Can be either "l2", "l1" or "robust"
    q : float, default: 0.4
        This parameter is used in robust loss for fine tuning. q < 1 gives less penalty to outliers
    eps : float, default: 0.01
        This parameter is a small constant used in robust loss to stabilize fine tuning.
    weights : list
        The weights to use for each scale
    average : str, default: "mean"
        The mode to set the average of the EPE map.
        If "mean", the mean of the EPE map is returned.
        If "sum", the EPE map is summed and divided by the batch size.
    resize_flow : str, default: "upsample"
        The mode to resize flow.
        If "upsample", predicted flow will be upsampled to the size of the ground truth.
        If "downsample", ground truth flow will be downsampled to the size of the predicted flow.
    extra_mask : torch.Tensor
        A mask to apply to the loss. Useful for removing the loss on the background
    use_valid_range : bool
        Whether to use the valid range of flow values for the loss
    valid_range : list
        The valid range of flow values for each scale
    """

    @configurable
    def __init__(
        self,
        norm="l1",
        q=0.4,
        eps=1e-2,
        weights=(1, 0.5, 0.25),
        average="mean",
        resize_flow="upsample",
        extra_mask=None,
        use_valid_range=True,
        valid_range=None,
    ):
        super(MultiScaleLoss, self).__init__()

        assert norm.lower() in (
            "l1",
            "l2",
            "robust",
        ), "Norm must be one of L1, L2, Robust"
        assert resize_flow.lower() in (
            "upsample",
            "downsample",
        ), "Resize flow must be one of upsample or downsample"
        assert average.lower() in ("mean", "sum"), "Average must be one of mean or sum"

        self.norm = norm.lower()
        self.q = q
        self.eps = eps
        self.weights = weights
        self.extra_mask = extra_mask
        self.use_valid_range = use_valid_range
        self.valid_range = valid_range
        self.average = average.lower()
        self.resize_flow = resize_flow.lower()

    @classmethod
    def from_config(cls, cfg):
        return {
            "norm": cfg.NORM,
            "weights": cfg.WEIGHTS,
            "average": cfg.AVERAGE,
            "resize_flow": cfg.RESIZE_FLOW,
            "extra_mask": cfg.EXTRA_MASK,
            "use_valid_range": cfg.USE_VALID_RANGE,
            "valid_range": cfg.VALID_RANGE,
        }

    def forward(self, pred, label):

        if label.shape[1] == 3:
            """Ignore valid mask for Multiscale Loss."""
            mask = label[:, 2:, :, :]
            label = label[:, :2, :, :]

        loss = 0
        b, c, h, w = label.size()

        if (
            (type(pred) is not tuple)
            and (type(pred) is not list)
            and (type(pred) is not set)
        ):
            pred = {pred}

        for i, level_pred in enumerate(pred):

            if self.resize_flow.lower() == "upsample":
                real_flow = F.interpolate(
                    level_pred, (h, w), mode="bilinear", align_corners=True
                )
                real_flow[:, 0, :, :] = real_flow[:, 0, :, :] * (
                    w / level_pred.shape[3]
                )
                real_flow[:, 1, :, :] = real_flow[:, 1, :, :] * (
                    h / level_pred.shape[2]
                )
                target = label

            elif self.resize_flow.lower() == "downsample":
                # down sample ground truth following irr solution
                # https://github.com/visinf/irr/blob/master/losses.py#L16
                b, c, h, w = level_pred.shape

                target = F.adaptive_avg_pool2d(label, [h, w])
                real_flow = level_pred

            if self.norm == "l2":
                loss_value = torch.norm(real_flow - target, p=2, dim=1)

            elif self.norm == "robust":
                loss_value = torch.norm(real_flow - target, p=1, dim=1)
                loss_value = (loss_value + self.eps) ** self.q

            elif self.norm == "l1":
                loss_value = torch.norm(real_flow - target, p=1, dim=1)

            if self.use_valid_range and self.valid_range is not None:

                with torch.no_grad():
                    mask = (target[:, 0, :, :].abs() <= self.valid_range[i][1]) & (
                        target[:, 1, :, :].abs() <= self.valid_range[i][0]
                    )
            else:
                with torch.no_grad():
                    mask = torch.ones(target[:, 0, :, :].shape).type_as(target)

            loss_value = loss_value * mask.float()

            if self.extra_mask is not None:
                val = self.extra_mask > 0
                loss_value = loss_value[val]

            if self.average.lower() == "mean":
                level_loss = loss_value.mean() * self.weights[i]

            elif self.average.lower() == "sum":
                level_loss = loss_value.sum() / b * self.weights[i]

            loss += level_loss

        loss = loss / len(pred)

        return loss
