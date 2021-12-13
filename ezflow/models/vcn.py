import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import Butterfly4D, SeparableConv4D, Soft4DFlowRegression
from ..encoder import build_encoder
from ..modules import conv
from ..utils import warp
from .build import MODEL_REGISTRY


def _gen_hypotheses_fusion_block(in_channels, out_channels):

    return nn.Sequential(
        *[
            conv(in_channels, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
            conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
            conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        ]
    )


@MODEL_REGISTRY.register()
class VCN(nn.Module):
    """
    Implementation of the paper
    `Volumetric Correspondence Networks for Optical Flow <https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(VCN, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)

        f_dim_a1 = cfg.DECODER.F_DIM_A1
        f_dim_a2 = cfg.DECODER.F_DIM_A2
        f_dim_b1 = cfg.DECODER.F_DIM_B1
        f_dim_b2 = cfg.DECODER.F_DIM_B2

        self.max_disps = cfg.MAX_DISPLACEMENTS
        self.factorization = cfg.FACTORIZATION

        self.butterfly_filters = nn.ModuleList()
        self.sep_conv_4d_filters = nn.ModuleList()

        for _ in range(3):

            self.butterfly_filters.append(
                Butterfly4D(
                    f_dim_a1,
                    f_dim_b1,
                    norm=cfg.DECODER.NORM,
                    full=False,
                )
            )
            self.sep_conv_4d_filters.append(
                SeparableConv4D(f_dim_b1, f_dim_b1, norm=False, full=False)
            )

        self.butterfly_filters.append(
            Butterfly4D(
                f_dim_a2,
                f_dim_b1,
                norm=cfg.DECODER.NORM,
                full=False,
            )
        )
        self.sep_conv_4d_filters.append(
            SeparableConv4D(f_dim_b1, f_dim_b1, norm=False, full=False)
        )

        self.butterfly_filters.append(
            Butterfly4D(
                f_dim_a2,
                f_dim_b2,
                norm=cfg.DECODER.NORM,
                full=True,
            )
        )
        self.sep_conv_4d_filters.append(
            SeparableConv4D(f_dim_b2, f_dim_b2, norm=False, full=True)
        )

        self.flow_regressors = nn.ModuleList()
        size = cfg.SIZE

        self.flow_regressors.append(
            Soft4DFlowRegression(
                [f_dim_b1 * size[0], size[1] // 64, size[2] // 64],
                max_disp=self.max_disps[0],
                entropy=cfg.DECODER.ENTROPY,
                factorization=self.factorization,
            )
        )

        scale = 32
        for i in range(1, 4):
            self.flow_regressors.append(
                Soft4DFlowRegression(
                    [
                        f_dim_b1 * size[0],
                        size[1] // scale,
                        size[2] // scale,
                    ],
                    max_disp=self.max_disps[i],
                    entropy=cfg.DECODER.ENTROPY,
                )
            )
            scale = scale // 2

        self.flow_regressors.append(
            Soft4DFlowRegression(
                [f_dim_b2 * size[0], size[1] // 4, size[2] // 4],
                max_disp=self.max_disps[0],
                entropy=cfg.DECODER.ENTROPY,
                factorization=self.factorization,
            )
        )

        self.hypotheses_fusion_blocks = nn.ModuleList()
        for i in range(1, 5):

            if i == 4:
                in_channels = 64 + (4 * i * f_dim_b1)
            else:
                in_channels = 128 + (4 * i * f_dim_b1)

            out_channels = 2 * i * f_dim_b1

            self.hypotheses_fusion_blocks.append(
                _gen_hypotheses_fusion_block(in_channels, out_channels)
            )

        self.hypotheses_fusion_blocks.append(
            _gen_hypotheses_fusion_block(
                64 + (16 * f_dim_b1) + (4 * f_dim_b2),
                (8 * f_dim_b1) + (2 * f_dim_b2),
            )
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

    def _corr_fn(self, features1, features2, max_disp, factorization=1):

        b, c, height, width = features1.shape

        if features1.is_cuda:
            cost = torch.cuda.FloatTensor(
                b,
                c,
                2 * max_disp + 1,
                2 * int(max_disp // factorization) + 1,
                height,
                width,
            ).fill_(0.0)
        else:
            cost = torch.FloatTensor(
                b,
                c,
                2 * max_disp + 1,
                2 * int(max_disp // factorization) + 1,
                height,
                width,
            ).fill_(0.0)

        for i in range(2 * max_disp + 1):
            ind = i - max_disp
            for j in range(2 * int(max_disp // factorization) + 1):
                indd = j - int(max_disp // factorization)
                feata = features1[
                    :, :, max(0, -indd) : height - indd, max(0, -ind) : width - ind
                ]
                featb = features2[
                    :, :, max(0, +indd) : height + indd, max(0, ind) : width + ind
                ]
                diff = feata * featb
                cost[
                    :,
                    :,
                    i,
                    j,
                    max(0, -indd) : height - indd,
                    max(0, -ind) : width - ind,
                ] = diff
        cost = F.leaky_relu(cost, 0.1, inplace=True)

        return cost

    def forward(self, img1, img2):

        batch_size = img1.shape[0]

        assert (
            batch_size == self.cfg.SIZE[0]
        ), f"Batch size in model configuration must be equal to the training batch size. Model config batch size: {self.cfg.SIZE[0]}, Training batch size: {batch_size}"

        # if self.cfg.SIZE[0] != img1.shape[0]:
        #     self.cfg.SIZE[0] = img1.shape[0]

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        for i in range(len(feature_pyramid1)):

            feature_pyramid1[i] = feature_pyramid1[i] / (
                torch.norm(feature_pyramid1[i], p=2, dim=1, keepdim=True) + 1e-9
            )
            feature_pyramid2[i] = feature_pyramid2[i] / (
                torch.norm(feature_pyramid2[i], p=2, dim=1, keepdim=True) + 1e-9
            )

        flow_preds = []
        flow_intermediates = []
        ent_intermediates = []
        scale = 32

        for i in range(len(self.butterfly_filters)):

            if i != 0:
                up_flow = (
                    F.interpolate(
                        flow_preds[-1],
                        [img1.shape[2] // scale, img1.shape[3] // scale],
                        mode="bilinear",
                        align_corners=True,
                    )
                    * 2
                )
                scale = scale // 2
                features2 = warp(feature_pyramid2[i], up_flow)

            else:
                features2 = feature_pyramid2[i]

            cost = self._corr_fn(
                feature_pyramid1[i],
                features2,
                self.max_disps[i],
                factorization=self.cfg.FACTORIZATION,
            )
            cost = self.butterfly_filters[i](cost)
            cost = self.sep_conv_4d_filters[i](cost)

            B, C, U, V, H, W = cost.shape
            cost = cost.view(-1, U, V, H, W)

            flow, ent = self.flow_regressors[i](cost)

            if i != 0:
                flow = flow.view(B, C, 2, H, W) + up_flow[:, np.newaxis]

            flow = flow.view(batch_size, -1, H, W)
            ent = ent.view(batch_size, -1, H, W)

            if i != 0:
                flow = torch.cat(
                    (
                        flow,
                        F.interpolate(
                            flow_intermediates[-1].detach() * 2,
                            [flow.shape[2], flow.shape[3]],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                    dim=1,
                )

                ent = torch.cat(
                    (
                        ent,
                        F.upsample(
                            ent_intermediates[-1],
                            [flow.shape[2], flow.shape[3]],
                            mode="bilinear",
                        ),
                    ),
                    dim=1,
                )

            flow_intermediates.append(flow)
            ent_intermediates.append(ent)

            x = torch.cat([ent.detach(), flow.detach(), feature_pyramid1[i]], dim=1)
            x = self.hypotheses_fusion_blocks[i](x)
            x = x.view(B, -1, 2, H, W)

            flow = (flow.view(B, -1, 2, H, W) * F.softmax(x, dim=1)).sum(dim=1)
            flow_preds.append(flow)

        flow_preds.reverse()

        scale = 4
        for i in range(len(flow_preds)):
            flow_preds[i] = F.interpolate(
                flow_preds[i],
                [img1.shape[2], img1.shape[3]],
                mode="bilinear",
                align_corners=True,
            )
            flow_preds[i] = flow_preds[i] * scale
            scale *= 2

        if self.training:
            return flow_preds

        return flow_preds[0]
