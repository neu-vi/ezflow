import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import FlowEntropy, build_decoder
from ..encoder import build_encoder
from ..modules import ConvNormRelu, build_module
from ..similarity import build_similarity
from ..utils import warp
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DICL(nn.Module):
    """
    Implementation of the paper
    `Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation <https://arxiv.org/abs/2010.14851>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(DICL, self).__init__()

        self.cfg = cfg
        self.context_net = cfg.CONTEXT_NET
        self.scale_factors = cfg.SCALE_FACTORS
        self.scale_contexts = cfg.SCALE_CONTEXTS

        self.feature_net = build_encoder(cfg.ENCODER)
        self.entropy_fn = FlowEntropy()

        matching_net = build_similarity(cfg.SIMILARITY.MATCHING_NET)

        search_range = cfg.SEARCH_RANGE

        self.cost_fn2 = build_similarity(
            cfg.SIMILARITY,
            max_u=search_range[0],
            max_v=search_range[0],
            matching_net=matching_net,
        )
        self.cost_fn3 = build_similarity(
            cfg.SIMILARITY,
            max_u=search_range[1],
            max_v=search_range[1],
            matching_net=matching_net,
        )
        self.cost_fn4 = build_similarity(
            cfg.SIMILARITY,
            max_u=search_range[2],
            max_v=search_range[2],
            matching_net=matching_net,
        )
        self.cost_fn5 = build_similarity(
            cfg.SIMILARITY,
            max_u=search_range[3],
            max_v=search_range[3],
            matching_net=matching_net,
        )
        self.cost_fn6 = build_similarity(
            cfg.SIMILARITY,
            max_u=search_range[4],
            max_v=search_range[4],
            matching_net=matching_net,
        )

        self.flow_decoder2 = build_decoder(
            cfg.DECODER, max_u=search_range[0], max_v=search_range[0]
        )
        self.flow_decoder3 = build_decoder(
            cfg.DECODER, max_u=search_range[1], max_v=search_range[1]
        )
        self.flow_decoder4 = build_decoder(
            cfg.DECODER, max_u=search_range[2], max_v=search_range[2]
        )
        self.flow_decoder5 = build_decoder(
            cfg.DECODER, max_u=search_range[3], max_v=search_range[3]
        )
        self.flow_decoder6 = build_decoder(
            cfg.DECODER, max_u=search_range[4], max_v=search_range[4]
        )

        if self.context_net:

            self.context_net2 = nn.Sequential(
                ConvNormRelu(38, 64, kernel_size=3, padding=1, dilation=1),
                ConvNormRelu(64, 128, kernel_size=3, padding=2, dilation=2),
                ConvNormRelu(128, 128, kernel_size=3, padding=4, dilation=4),
                ConvNormRelu(128, 96, kernel_size=3, padding=8, dilation=8),
                ConvNormRelu(96, 64, kernel_size=3, padding=16, dilation=16),
                ConvNormRelu(64, 32, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.context_net3 = nn.Sequential(
                ConvNormRelu(38, 64, kernel_size=3, padding=1, dilation=1),
                ConvNormRelu(64, 128, kernel_size=3, padding=2, dilation=2),
                ConvNormRelu(128, 128, kernel_size=3, padding=4, dilation=4),
                ConvNormRelu(128, 96, kernel_size=3, padding=8, dilation=8),
                ConvNormRelu(96, 64, kernel_size=3, padding=16, dilation=16),
                ConvNormRelu(64, 32, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.context_net4 = nn.Sequential(
                ConvNormRelu(38, 64, kernel_size=3, padding=1, dilation=1),
                ConvNormRelu(64, 128, kernel_size=3, padding=2, dilation=2),
                ConvNormRelu(128, 128, kernel_size=3, padding=4, dilation=4),
                ConvNormRelu(128, 64, kernel_size=3, padding=8, dilation=8),
                ConvNormRelu(64, 32, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.context_net5 = nn.Sequential(
                ConvNormRelu(38, 64, kernel_size=3, padding=1, dilation=1),
                ConvNormRelu(64, 128, kernel_size=3, padding=2, dilation=2),
                ConvNormRelu(128, 64, kernel_size=3, padding=4, dilation=4),
                ConvNormRelu(64, 32, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.context_net6 = nn.Sequential(
                ConvNormRelu(38, 64, kernel_size=3, padding=1, dilation=1),
                ConvNormRelu(64, 64, kernel_size=3, padding=2, dilation=2),
                ConvNormRelu(64, 32, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True),
            )

        self._init_weights()

        if cfg.DAP.USE_DAP:

            name = "DisplacementAwareProjection"

            self.dap_layer2 = build_module(
                cfg.DAP, name=name, max_displacement=search_range[0]
            )
            self.dap_layer3 = build_module(
                cfg.DAP, name=name, max_displacement=search_range[1]
            )
            self.dap_layer4 = build_module(
                cfg.DAP, name=name, max_displacement=search_range[2]
            )
            self.dap_layer5 = build_module(
                cfg.DAP, name=name, max_displacement=search_range[3]
            )
            self.dap_layer6 = build_module(
                cfg.DAP, name=name, max_displacement=search_range[4]
            )

            if cfg.DAP.INIT_ID:
                nn.init.eye_(
                    self.dap_layer2.dap_layer.conv.weight.squeeze(-1).squeeze(-1)
                )
                nn.init.eye_(
                    self.dap_layer3.dap_layer.conv.weight.squeeze(-1).squeeze(-1)
                )
                nn.init.eye_(
                    self.dap_layer4.dap_layer.conv.weight.squeeze(-1).squeeze(-1)
                )
                nn.init.eye_(
                    self.dap_layer5.dap_layer.conv.weight.squeeze(-1).squeeze(-1)
                )
                nn.init.eye_(
                    self.dap_layer6.dap_layer.conv.weight.squeeze(-1).squeeze(-1)
                )

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, (nn.Conv2d)) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _process_level(
        self,
        x,
        y,
        orig_img,
        level,
        prev_upflow,
        scale_factor,
        upflow_size,
        scale_context=None,
        warp_flow=True,
    ):

        level = str(level)

        cost_fn = getattr(self, "cost_fn" + level)
        flow_decoder = getattr(self, "flow_decoder" + level)
        dap_layer = getattr(self, "dap_layer" + level)

        if self.context_net:
            context_net = getattr(self, "context_net" + level)

        if warp_flow:
            warped_flow = warp(y, prev_upflow)
            cost = cost_fn(x, warped_flow)
        else:
            cost = cost_fn(x, y)

        g = F.interpolate(
            orig_img,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=True,
        )

        if self.cfg.DAP.USE_DAP:
            cost = dap_layer(cost)

        if warp_flow:
            flow = flow_decoder(cost) + prev_upflow
        else:
            flow = flow_decoder(cost)

        if self.context_net:

            if self.cfg.SUP_RAW_FLOW:
                raw_flow = flow
            else:
                raw_flow = None

            entropy = self.entropy_fn(cost)
            features = torch.cat((flow.detach(), entropy.detach(), x, g), dim=1)
            flow = flow + context_net(features) * scale_context

        upflow = 2.0 * F.interpolate(
            flow, upflow_size, mode="bilinear", align_corners=True
        )
        upflow = upflow.detach()

        return upflow, flow, raw_flow

    def forward(self, img1, img2):
        """
        Performs forward pass of the network

        Parameters
        ----------
        img1 : torch.Tensor
            Image to predict flow from
        img2 : torch.Tensor
            Image to predict flow to

        Returns
        -------
        torch.Tensor
            Flow from img1 to img2
        """

        _, x2, x3, x4, x5, x6 = self.feature_net(img1)
        _, y2, y3, y4, y5, y6 = self.feature_net(img2)

        upflow6, flow6, raw_flow6 = self._process_level(
            x6,
            y6,
            img1,
            6,
            None,
            self.scale_factors[4],
            (x5.shape[2], x5.shape[3]),
            self.scale_contexts[4],
            warp_flow=False,
        )

        upflow5, flow5, raw_flow5 = self._process_level(
            x5,
            y5,
            img1,
            5,
            upflow6,
            self.scale_factors[3],
            (x4.shape[2], x4.shape[3]),
            self.scale_contexts[3],
        )

        upflow4, flow4, raw_flow4 = self._process_level(
            x4,
            y4,
            img1,
            4,
            upflow5,
            self.scale_factors[2],
            (x3.shape[2], x3.shape[3]),
            self.scale_contexts[2],
        )

        upflow3, flow3, raw_flow3 = self._process_level(
            x3,
            y3,
            img1,
            3,
            upflow4,
            self.scale_factors[1],
            (x2.shape[2], x2.shape[3]),
            self.scale_contexts[1],
        )

        _, flow2, raw_flow2 = self._process_level(
            x2,
            y2,
            img1,
            2,
            upflow3,
            self.scale_factors[0],
            (x2.shape[2], x2.shape[3]),
            self.scale_contexts[0],
        )

        if self.training:

            if self.cfg.SUP_RAW_FLOW:
                return (
                    flow2,
                    raw_flow2,
                    flow3,
                    raw_flow3,
                    flow4,
                    raw_flow4,
                    flow5,
                    raw_flow5,
                    flow6,
                    raw_flow6,
                )

            return (flow2, flow3, flow4, flow5, flow6)

        else:
            _, _, height, width = img1.size()
            return F.interpolate(
                flow2, (height, width), mode="bilinear", align_corners=True
            )
