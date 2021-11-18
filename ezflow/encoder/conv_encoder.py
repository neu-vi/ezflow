import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, norm=None):

    if norm == "group":
        norm_fn = nn.GroupNorm(num_groups=8, num_channels=out_planes)

    elif norm == "batch":
        norm_fn = nn.BatchNorm2d(out_planes)

    elif norm == "instance":
        norm_fn = nn.InstanceNorm2d(out_planes)

    else:
        norm_fn = nn.Identity()

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        ),
        norm_fn,
        nn.LeakyReLU(0.1, inplace=True),
    )


@ENCODER_REGISTRY.register()
class FlownetConvEncoder(nn.Module):
    """Convolution encoder for FlowNet"""

    @configurable
    def __init__(
        self,
        in_channels=3,
        config=[64, 128, 256, 512],
        norm=None,
    ):
        super(FlownetConvEncoder, self).__init__()

        assert (
            len(config) >= 3
        ), "FlownetConvEncoder expects at least 3 output channels in config."

        if isinstance(config, tuple):
            config = list(config)

        channels = [in_channels] + config

        self.encoder = nn.ModuleList()
        self.encoder.append(conv(channels[0], channels[1], kernel_size=7, stride=2))
        self.encoder.append(conv(channels[1], channels[2], kernel_size=5, stride=2))
        self.encoder.append(conv(channels[2], channels[3], kernel_size=5, stride=2))

        channels = channels[3:]

        for i in range(len(channels) - 1):

            stride = 1 if i % 2 == 0 else 2

            self.encoder.append(
                conv(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=stride,
                    norm=norm,
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "config": cfg.CONFIG,
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
