import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import configurable
from .base_module import BaseModule
from .build import MODULE_REGISTRY, build_module


def _get_norm_fn(in_dim, norm="instance"):
    assert norm in ["instance", "batch", "none"]
    if norm == "instance":
        return nn.InstanceNorm2d(in_dim)
    elif norm == "batch":
        return nn.BatchNorm2d(in_dim)
    elif norm == "none":
        return nn.Identity()


@MODULE_REGISTRY.register()
class DCVFilterGroupConvStemJoint_SingelStage(BaseModule):
    @configurable
    def __init__(
        self,
        unet_cfg,
        cv_num_groups,
        num_dilations,
        cv_search_range,
        feat_in_planes,
        out_channels,
        hidden_dim,
        use_cost_volume_residual,
        use_stem_group_conv,
        norm,
    ):
        super(DCVFilterGroupConvStemJoint_SingelStage, self).__init__()

        in_channels = cv_num_groups * num_dilations * (cv_search_range**2)
        self.use_cost_volume_residual = use_cost_volume_residual

        stem_num_groups = num_dilations
        if not use_stem_group_conv:
            stem_num_groups = 1

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                2 * in_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=stem_num_groups,
            ),
            _get_norm_fn(2 * in_channels, norm),
            nn.ReLU(),
            nn.Conv2d(
                2 * in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=stem_num_groups,
            ),
            _get_norm_fn(in_channels, norm),
            nn.ReLU(),
        )

        self.stem_xform = nn.Identity()
        if in_channels != out_channels:
            self.stem_xform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU()
            )

        unet_cfg.IN_CHANNELS = out_channels + feat_in_planes
        unet_cfg.HIDDEN_DIM = hidden_dim
        unet_cfg.OUT_CHANNELS = out_channels

        self.unet = build_module(unet_cfg)

        self.flow_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.up_out = nn.Conv2d(hidden_dim, 8**2 * 9, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "unet_cfg": cfg.UNET,
            "cv_num_groups": cfg.GROUPS,
            "num_dilations": cfg.NUM_DILATIONS,
            "cv_search_range": cfg.COST_VOLUME_SEARCH_RANGE,
            "feat_in_planes": cfg.FEAT_IN_PLANES,
            "out_channels": cfg.OUT_CHANNELS,
            "hidden_dim": cfg.HIDDEN_DIM,
            "use_cost_volume_residual": cfg.USE_COST_VOLUME_RESIDUAL,
            "use_stem_group_conv": cfg.USE_STEM_GROUP_CONV,
            "norm": cfg.NORM,
        }

    def forward(self, cost, x):
        # we use the feature from the stride of 8 only for now
        x = x[-1]

        b, c, u, v, h, w = cost.shape
        cost = cost.view(b, c * u * v, h, w)

        cost = self.stem(cost)
        cost = self.stem_xform(cost)
        flow_logits_stage0 = cost

        x_out = torch.cat((x, cost), dim=1)
        x_out = self.unet(x_out)
        flow_logits_stage1 = self.flow_out(x_out)

        if self.use_cost_volume_residual:
            flow_logits_stage1 = flow_logits_stage1 + flow_logits_stage0

        up_logits_stage1 = self.up_out(x_out)

        return (flow_logits_stage0, flow_logits_stage1), (None, up_logits_stage1)
