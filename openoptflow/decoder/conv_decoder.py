import torch
import torch.nn as nn

from ..config import configurable
from .build import DECODER_REGISTRY


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


@DECODER_REGISTRY.register()
class ConvDecoder(nn.Module):
    """Convolutional decoder"""

    @configurable
    def __init__(
        self,
        config=[128, 128, 96, 64, 32],
        concat_channels=None,
        to_flow=True,
        block=None,
    ):
        super().__init__()

        self.concat_channels = concat_channels

        if block is None:
            block = conv

        self.decoder = nn.ModuleList()
        config_cumsum = torch.cumsum(torch.tensor(config), dim=0)

        if concat_channels is not None:
            self.decoder.append(
                block(concat_channels, config[0], kernel_size=3, stride=1)
            )

        for i in range(len(config) - 1):

            if concat_channels is not None:
                in_channels = config_cumsum[i] + concat_channels
            else:
                in_channels = config[i]

            self.decoder.append(
                block(in_channels, config[i + 1], kernel_size=3, stride=1)
            )

        self.to_flow = nn.Identity()

        if to_flow:

            if concat_channels is not None:
                in_channels = config_cumsum[-1] + concat_channels
            else:
                in_channels = config[-1]

            self.to_flow = nn.Conv2d(
                in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
            )

    @classmethod
    def from_config(self, cfg):
        return {"config": cfg.CONFIG}

    def forward(self, x):

        for i in range(len(self.decoder)):

            y = self.decoder[i](x)

            if self.concat_channels is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y

        return self.to_flow(x), x
