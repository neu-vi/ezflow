import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


@ENCODER_REGISTRY.register()
class PyramidEncoder(nn.Module):

    """Pyramid encoder with hierarchy of features"""

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
