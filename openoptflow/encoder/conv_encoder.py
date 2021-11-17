import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, batch_norm=False):
    if batch_norm == "batch":
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
        out_channels=[64, 128, 256, 512],
        batch_norm=False,
    ):
        super(ConvEncoder, self).__init__()

        assert len(out_channels) >= 3, "encoder expects at least 3 out channels."

        if isinstance(out_channels, tuple):
            out_channels = list(out_channels)

        channels = [in_channels] + out_channels

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
                    batch_norm=batch_norm,
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "batch_norm": cfg.BATCH_NORM,
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
