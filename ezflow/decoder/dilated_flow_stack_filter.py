import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import configurable
from ..modules import BaseModule, build_module
from ..utils import convex_upsample_flow
from .build import DECODER_REGISTRY, build_decoder


def _get_norm_fn(in_dim, norm="instance"):
    assert norm in ["instance", "batch", "none"]
    if norm == "instance":
        return nn.InstanceNorm2d(in_dim)
    elif norm == "batch":
        return nn.BatchNorm2d(in_dim)
    elif norm == "none":
        return nn.Identity()


@DECODER_REGISTRY.register()
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


@DECODER_REGISTRY.register()
class DCVDilatedFlowStackFilterDecoder(BaseModule):
    @configurable
    def __init__(self, cost_volume_filter_cfg, feat_strides, dilations):
        super(DCVDilatedFlowStackFilterDecoder, self).__init__()

        self.feat_strides = feat_strides

        self.num_dilations = 0
        for dilations_i in dilations:
            self.num_dilations += len(dilations_i)

        cost_volume_filter_cfg.NUM_DILATIONS = self.num_dilations
        cost_volume_filter_cfg.OUT_CHANNELS = self.num_dilations * (
            cost_volume_filter_cfg.SEARCH_RANGE**2
        )

        self.cost_volume_filter = build_decoder(cost_volume_filter_cfg)

        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                print(
                    "\n************** Attention: BN layers found in the deocder. ******************\n"
                )

    @classmethod
    def from_config(cls, cfg):
        return {
            "cost_volume_filter_cfg": cfg.COST_VOLUME_FILTER,
            "feat_strides": cfg.FEAT_STRIDES,
            "dilations": cfg.DILATIONS,
        }

    def logits_to_flow(self, flow_logits, flow_offsets):
        b, cuv, h, w = flow_logits.shape

        flow_logits = flow_logits.view(b, self.num_dilations, -1, h, w)

        flow_logits = flow_logits.view(b, -1, h, w)
        flow_probs = F.softmax(flow_logits, 1)
        flow_y = torch.sum(flow_probs * flow_offsets[:, :, 0], dim=1)
        flow_x = torch.sum(flow_probs * flow_offsets[:, :, 1], dim=1)
        flow = torch.stack((flow_x, flow_y), dim=1)
        flow_entropy = -flow_probs * torch.clamp(flow_probs, 1e-9, 1 - 1e-9).log()
        flow_entropy = flow_entropy.sum(1, keepdim=True)

        return flow, flow_entropy

    def interpolate_flow(self, flow, up_mask_logits):
        if up_mask_logits is not None:
            return convex_upsample_flow(flow, up_mask_logits, self.feat_strides[-1])

        return F.interpolate(flow, scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, cost, context, flow_offsets):
        flow_logits_list, up_mask_logits_list = self.cost_volume_filter(cost, context)
        flow_list = []

        if self.training:
            for idx, flow_logits_i in enumerate(flow_logits_list):
                flow_i, _ = self.logits_to_flow(flow_logits_i, flow_offsets)
                flow_i = self.interpolate_flow(flow_i, up_mask_logits_list[idx])
                flow_list.append(flow_i)

            return flow_list, flow_logits_list

        flow_list = [None for _ in flow_logits_list]
        flow = self.logits_to_flow(flow_logits_list[-1], flow_offsets)
        flow = self.interpolate_flow(flow, up_mask_logits_list[-1])
        flow_list[-1] = flow

        return flow_list, flow_logits_list
