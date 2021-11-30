import torch.nn as nn

from ..config import configurable
from ..modules import conv
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class PyramidEncoder(nn.Module):
    """
    Pyramid encoder which returns a hierarchy of features
    Used in **PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume** (https://arxiv.org/abs/1709.02371)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration of the pyramid encoder's layers
    """

    @configurable
    def __init__(self, in_channels=3, config=[16, 32, 64, 96, 128, 196]):
        super().__init__()

        if isinstance(config, tuple):
            config = list(config)
        config = [in_channels] + config

        self.encoder = nn.ModuleList()

        for i in range(len(config) - 1):
            self.encoder.append(
                nn.Sequential(
                    conv(config[i], config[i + 1], kernel_size=3, stride=2),
                    conv(config[i + 1], config[i + 1], kernel_size=3, stride=1),
                    conv(config[i + 1], config[i + 1], kernel_size=3, stride=1),
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "config": cfg.CONFIG,
        }

    def forward(self, img):

        feature_pyramid = []
        x = img

        for i in range(len(self.encoder)):

            x = self.encoder[i](x)
            feature_pyramid.append(x)

        feature_pyramid.reverse()

        return feature_pyramid
