import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, norm=None):
    if norm == "batch":
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


@ENCODER_REGISTRY.register()
class ConvEncoder(nn.Module):
    """Convolution encoder"""

    @configurable
    def __init__(
        self,
        in_channels=3,
        channels=[64, 128, 256],
        kernels=[3, 3, 3],
        strides=[1, 1, 1],
        norm=None,
    ):
        super(ConvEncoder, self).__init__()

        assert len(channels) == len(
            kernels
        ), "number of kernels and channels are not equal"
        assert len(channels) == len(
            strides
        ), "number of strides and channels are not equal"

        norm = norm if norm is None else norm.lower()

        if isinstance(channels, tuple):
            channels = list(channels)
        if isinstance(kernels, tuple):
            kernels = list(kernels)
        if isinstance(strides, tuple):
            strides = list(strides)

        channels = [in_channels] + channels

        self.encoder = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.encoder.append(
                conv(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    norm=norm,
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "channels": cfg.LAYER_CONFIG.CHANNELS,
            "kernels": cfg.LAYER_CONFIG.KERNELS,
            "strides": cfg.LAYER_CONFIG.STRIDES,
            "norm": cfg.NORM,
        }

    def forward(self, x):

        outputs = []

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

            if len(outputs) > 0:
                prev_output = outputs[-1]
                if prev_output.shape[1:] == x.shape[1:]:
                    outputs[-1] = x
                else:
                    outputs.append(x)
            else:
                outputs.append(x)

        return outputs
