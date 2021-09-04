import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        norm=None,
        activation="relu",
        **kwargs
    ):
        super(ConvNormRelu, self).__init__()

        if norm.lower() == "group":
            self.norm = nn.GroupNorm(out_channels)

        elif norm.lower() == "batch":
            self.norm = nn.BatchNorm2d(out_channels)

        elif norm.lower() == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)

        else:
            self.norm = nn.Sequential()

        if activation.lower() == "leakyrelu":
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        if deconv:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, bias=False, **kwargs
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x