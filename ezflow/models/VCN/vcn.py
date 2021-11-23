import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import Butterfly4D, ConvDecoder, SeparableConv4D
from ...encoder import build_encoder
from ...similarity import IterSpatialCorrelationSampler
from ...utils import warp
from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VCN(nn.Module):
    def __init__(self, cfg):
        super(VCN, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)

        self.butterfly_filters = nn.ModuleList()
        self.sep_conv_4d_filters = nn.ModuleList()

        for _ in range(3):

            self.butterfly_filters.append(
                Butterfly4D(
                    cfg.DECODER.F_DIM_A1,
                    cfg.DECODER.F_DIM_B1,
                    norm=cfg.DECODER.NORM,
                    full=False,
                )
            )
            self.sep_conv_4d_filters.append(
                SeparableConv4D(
                    cfg.DECODER.F_DIM_B1, cfg.DECODER.F_DIM_B1, norm=False, full=False
                )
            )

        self.butterfly_filters.append(
            Butterfly4D(
                cfg.DECODER.F_DIM_A2,
                cfg.DECODER.F_DIM_B1,
                norm=cfg.DECODER.NORM,
                full=False,
            )
        )
        self.sep_conv_4d_filters.append(
            SeparableConv4D(
                cfg.DECODER.F_DIM_B1, cfg.DECODER.F_DIM_B1, norm=False, full=False
            )
        )

        self.butterfly_filters.append(
            Butterfly4D(
                cfg.DECODER.F_DIM_A2,
                cfg.DECODER.F_DIM_B2,
                norm=cfg.DECODER.NORM,
                full=True,
            )
        )
        self.sep_conv_4d_filters.append(
            SeparableConv4D(
                cfg.DECODER.F_DIM_B2, cfg.DECODER.F_DIM_B2, norm=False, full=True
            )
        )

    def forward(self, img1, img2):

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        for i in range(len(feature_pyramid1)):

            feature_pyramid1[i] = feature_pyramid1[i] / (
                torch.norm(feature_pyramid1[i], p=2, dim=1, keepdim=True) + 1e-9
            )
            feature_pyramid2[i] = feature_pyramid2[i] / (
                torch.norm(feature_pyramid2[i], p=2, dim=1, keepdim=True) + 1e-9
            )
