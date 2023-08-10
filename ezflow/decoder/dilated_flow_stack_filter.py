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
    def __init__(self, cost_volume_filter_cfg, feat_strides, flow_offsets, norm):
        super(DCVDilatedFlowStackFilterDecoder, self).__init__()

        self.feat_strides = feat_strides

        dilations = cost_volume_filter_cfg.DILATIONS
        num_dilations = 0
        for dilations_i in dilations:
            num_dilations += len(dilations_i)
        self.num_dilations = num_dilations
        cost_volume_dim = (
            cost_volume_filter_cfg.NUM_GROUPS
            * num_dilations
            * (cost_volume_filter_cfg.SEARCH_RANGE**2)
        )
        inter_dim = cost_volume_filter_cfg.UNET.HIDDEN_DIM
        cost_volume_filter_cfg.OUT_CHANNELS = num_dilations * (
            cost_volume_filter_cfg.SEARCH_RANGE**2
        )

        self.cost_volume_filter = build_module(cost_volume_filter_cfg.NAME)

        flow_offsets = flow_offsets.view(1, -1, 2, 1, 1)
        self.register_buffer("flow_offsets", flow_offsets)

        found_bn = False
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                found_bn = True
        if found_bn:
            print(
                "\n************** Attention: BN layers found in the deocder. ******************\n"
            )

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

    def forward(self, cost, x1):
        flow_logits_list, up_mask_logits_list = self.cost_volume_filter(cost, x1)
        flow_list = []
        if self.training:
            for idx, flow_logits_i in enumerate(flow_logits_list):
                if flow_logits_i is None:
                    flow_list.append(None)
                else:
                    flow_i, _ = self.logits_to_flow(flow_logits_i)
                    if up_mask_logits_list[idx] is not None:
                        flow_i = convex_upsample_flow(
                            flow_i, up_mask_logits_list[idx], self.feat_strides[-1]
                        )
                    else:
                        flow_i = F.interpolate(flow_i, scale_factor=8, mode="bilinear")
                    flow_list.append(flow_i)

            return flow_list, flow_logits_list

        flow_list = [None for _ in flow_logits_list]
        if flow_logits_list[-1] is not None:
            flow_i, _ = self.logits_to_flow(flow_logits_list[-1])
            flow_i = convex_upsample_flow(
                flow_i, up_mask_logits_list[-1], self.feat_strides[-1]
            )
        else:
            flow_i, _ = self.logits_to_flow(flow_logits_list[-2])
            flow_i = convex_upsample_flow(
                flow_i, up_mask_logits_list[-2], self.feat_strides[-1]
            )
        flow_list[-1] = flow_i

        return flow_list, flow_logits_list
