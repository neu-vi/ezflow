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


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


@DECODER_REGISTRY.register()
class ConvDecoder(nn.Module):
    """Convolutional decoder"""

    @configurable
    def __init__(
        self, config=[128, 128, 96, 64, 32], concat_channels=None, to_flow=True
    ):
        super().__init__()

        self.concat_channels = concat_channels

        self.decoder = nn.ModuleList()
        config_cumsum = torch.cumsum(torch.tensor(config), dim=0)

        if concat_channels is not None:
            self.decoder.append(
                conv(concat_channels, config[0], kernel_size=3, stride=1)
            )

        for i in range(len(config) - 1):

            if concat_channels is not None:
                in_channels = config_cumsum[i] + concat_channels
            else:
                in_channels = config[i]

            self.decoder.append(
                conv(in_channels, config[i + 1], kernel_size=3, stride=1)
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


@DECODER_REGISTRY.register()
class ConvFlowDecoder(nn.Module):
    """Convolutional decoder to regress and upsample the optical flow"""

    @configurable
    def __init__(self, in_channels=1024, channels=[512, 256, 128, 64]):
        super().__init__()

        if isinstance(channels, tuple):
            channels = list(channels)

        out_channels = [in_channels] + channels
        in_channels = []
        prev_out_channels = 0
        for i in range(len(out_channels)):
            if i > 0:
                inp = out_channels[i] + prev_out_channels + 2
                prev_out_channels = out_channels[i]
            else:
                inp = out_channels[i]
                prev_out_channels = out_channels[i + 1]

            in_channels.append(inp)

        self.predict_flow = nn.ModuleList()
        self.upsample_flow = nn.ModuleList()
        self.deconv = nn.ModuleList()

        print("in_channels:", in_channels)
        for i in range(len(in_channels) - 1):
            self.predict_flow.append(
                nn.Conv2d(
                    in_channels[i], 2, kernel_size=3, stride=1, padding=1, bias=False
                ),
            )

            self.upsample_flow.append(
                nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
            )

            self.deconv.append(deconv(in_channels[i], out_channels[i + 1]))

        self.to_flow = nn.Conv2d(
            in_channels[-1], 2, kernel_size=3, stride=1, padding=1, bias=False
        )

    @classmethod
    def from_config(self, cfg):
        return {"in_channels": cfg.IN_CHANNELS, "channels": cfg.LAYER_CONFIG.CHANNELS}

    def forward(self, x):
        flow_preds = []

        conv_out = x[-1]

        flow = self.predict_flow[0](conv_out)
        flow_up = self.upsample_flow[0](flow)
        deconv_out = self.deconv[0](conv_out)

        flow_preds.append(flow)

        layer_index = 1

        start = len(x) - 2
        end = 1

        for conv_out in x[start:end:-1]:
            assert conv_out.shape[2] == deconv_out.shape[2] == flow_up.shape[2]
            assert conv_out.shape[3] == deconv_out.shape[3] == flow_up.shape[3]

            concat_out = torch.cat((conv_out, deconv_out, flow_up), dim=1)

            flow = self.predict_flow[layer_index](concat_out)
            flow_up = self.upsample_flow[layer_index](flow)
            deconv_out = self.deconv[layer_index](concat_out)

            flow_preds.append(flow)

            layer_index += 1

        concat_out = torch.cat((x[1], deconv_out, flow_up), dim=1)
        flow = self.to_flow(concat_out)
        flow_preds.append(flow)

        return flow_preds
