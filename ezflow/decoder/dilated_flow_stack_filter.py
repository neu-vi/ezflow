import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import configurable
from ..modules import BaseModule, build_module
from ..utils import convex_upsample_flow
from .build import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class DCVDilatedFlowStackFilterDecoder(nn.Module):
    @configurable
    def __init__(self, cost_volume_filter_cfg, feat_strides, dilations, flow_offsets):
        super(DCVDilatedFlowStackFilterDecoder, self).__init__()

        self.feat_strides = feat_strides

        self.num_dilations = 0
        for dilations_i in dilations:
            self.num_dilations += len(dilations_i)

        flow_offsets = flow_offsets.view(1, -1, 2, 1, 1)
        self.register_buffer("flow_offsets", flow_offsets)

        cost_volume_filter_cfg.NUM_DILATIONS = self.num_dilations
        cost_volume_filter_cfg.OUT_CHANNELS = self.num_dilations * (
            cost_volume_filter_cfg.SEARCH_RANGE**2
        )

        self.cost_volume_filter = build_module(cost_volume_filter_cfg.NAME)

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
            "flow_offsets": cfg.FLOW_OFFSETS,
        }

    def get_flow_offsets(self, search_radius, feat_strides, dilation):
        x_grids = np.arange(-search_radius, search_radius + 1) * dilation * feat_strides
        grids = np.meshgrid(x_grids, x_grids)
        grids = np.stack(grids, 2)
        return grids.reshape(1, (2 * search_radius + 1) ** 2 * 2, 1, 1)

    def logits_to_flow(self, flow_logits):
        b, cuv, h, w = flow_logits.shape
        # assert self.num_hypothesis == 1
        flow_logits = flow_logits.view(b, self.num_dilations, -1, h, w)

        flow_logits = flow_logits.view(b, -1, h, w)
        flow_probs = F.softmax(flow_logits, 1)
        flow_y = torch.sum(flow_probs * self.flow_offsets[:, :, 0], dim=1)
        flow_x = torch.sum(flow_probs * self.flow_offsets[:, :, 1], dim=1)
        flow = torch.stack((flow_x, flow_y), dim=1)
        flow_entropy = -flow_probs * torch.clamp(flow_probs, 1e-9, 1 - 1e-9).log()
        flow_entropy = flow_entropy.sum(1, keepdim=True)

        return flow, flow_entropy

    def interpolate_flow(self, flow, up_mask_logits):
        if up_mask_logits:
            return convex_upsample_flow(flow, up_mask_logits, self.feat_strides[-1])

        return F.interpolate(flow, scale_factor=8, mode="bilinear")

    def forward(self, cost, x):
        flow_logits_list, up_mask_logits_list = self.cost_volume_filter(cost, x)
        flow_list = []

        if self.training:
            for idx, flow_logits_i in enumerate(flow_logits_list):
                flow_i, _ = self.logits_to_flow(flow_logits_i)
                flow_i = self.interpolate_flow(flow_i, up_mask_logits_list[idx])
                flow_list.append(flow_i)

            return flow_list, flow_logits_list

        flow_list = [None for _ in flow_logits_list]
        flow = self.logits_to_flow(flow_logits_list[-1])
        flow = self.interpolate_flow(flow, up_mask_logits_list[-1])
        flow_list[-1] = flow

        return flow_list, flow_logits_list
