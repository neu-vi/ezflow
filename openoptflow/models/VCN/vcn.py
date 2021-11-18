import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import ConvDecoder
from ...encoder import build_encoder
from ...similarity import CorrelationLayer, IterSpatialCorrelationSampler
from ...utils import warp
from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VCN(nn.Module):
    def __init__(self, cfg):
        super(VCN, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg)

        self.butterfly_filters = nn.ModuleList()

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
