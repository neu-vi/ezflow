import torch
import torch.nn as nn

from ..config import configurable
from ..modules import conv
from .build import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class ContextNetwork(nn.Module):
    """
    PWCNet Context Network decoder

    """

    @configurable
    def __init__(self, in_channels=565, config=[128, 128, 96, 64, 32]):
        super(ContextNetwork, self).__init__()

        self.context_net = nn.ModuleList(
            [
                conv(
                    in_channels,
                    config[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                ),
            ]
        )
        self.context_net.append(
            conv(config[0], config[0], kernel_size=3, stride=1, padding=2, dilation=2)
        )
        self.context_net.append(
            conv(config[0], config[1], kernel_size=3, stride=1, padding=4, dilation=4)
        )
        self.context_net.append(
            conv(config[1], config[2], kernel_size=3, stride=1, padding=8, dilation=8)
        )
        self.context_net.append(
            conv(config[2], config[3], kernel_size=3, stride=1, padding=16, dilation=16)
        )
        self.context_net.append(
            conv(config[3], config[4], kernel_size=3, stride=1, padding=1, dilation=1)
        )
        self.context_net.append(
            nn.Conv2d(config[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.context_net = nn.Sequential(*self.context_net)

    @classmethod
    def from_config(self, cfg):
        return {"in_channels": cfg.IN_CHANNELS, "config": cfg.CONFIG}

    def forward(self, x):
        return self.context_net(x)
