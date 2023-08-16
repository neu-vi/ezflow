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
class DCVFilterGroupConvStemJoint(BaseModule):
    @configurable
    def __init__(
        self,
        unet_cfg,
        num_groups,
        num_dilations,
        search_range,
        feat_in_planes,
        out_channels,
        hidden_dim,
        use_filter_residual,
        use_group_conv_stem,
        norm,
    ):
        super(DCVFilterGroupConvStemJoint, self).__init__()

        in_channels = num_groups * num_dilations * (search_range**2)
        self.use_filter_residual = use_filter_residual

        stem_num_groups = num_dilations
        if not use_group_conv_stem:
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
        unet_cfg.OUT_CHANNELS = hidden_dim

        self.unet = build_module(unet_cfg)

        self.flow_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.up_out = nn.Conv2d(hidden_dim, 8**2 * 9, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "unet_cfg": cfg.UNET,
            "num_groups": cfg.NUM_GROUPS,
            "num_dilations": cfg.NUM_DILATIONS,
            "search_range": cfg.SEARCH_RANGE,
            "feat_in_planes": cfg.FEAT_IN_PLANES,
            "out_channels": cfg.OUT_CHANNELS,
            "hidden_dim": cfg.HIDDEN_DIM,
            "use_filter_residual": cfg.USE_FILTER_RESIDUAL,
            "use_group_conv_stem": cfg.USE_GROUP_CONV_STEM,
            "norm": cfg.NORM,
        }

    def forward(self, cost, context_fmap):
        # use the feature from the stride of 8
        context_fmap = context_fmap[-1]

        b, c, u, v, h, w = cost.shape
        cost = cost.view(b, c * u * v, h, w)

        cost = self.stem(cost)
        flow_logits_stage0 = self.stem_xform(cost)

        out = torch.cat((context_fmap, cost), dim=1)
        out = self.unet(out)
        flow_logits_stage1 = self.flow_out(out)

        if self.use_filter_residual:
            flow_logits_stage1 = flow_logits_stage1 + flow_logits_stage0

        up_logits_stage1 = self.up_out(out)

        return (flow_logits_stage0, flow_logits_stage1), (None, up_logits_stage1)
