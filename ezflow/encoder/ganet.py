import torch
import torch.nn as nn

from ..config import configurable
from ..modules import Conv2x, ConvNormRelu
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class GANetBackbone(nn.Module):
    """
    Feature extractor backbone used in **GA-Net: Guided Aggregation Net for End-to-end Stereo Matching** (https://arxiv.org/abs/1904.06587)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """

    @configurable
    def __init__(self, in_channels=3, out_channels=32):
        super(GANetBackbone, self).__init__()

        self.conv_start = nn.Sequential(
            ConvNormRelu(in_channels, 32, kernel_size=3, padding=1),
            ConvNormRelu(32, 32, kernel_size=3, stride=2, padding=1),
            ConvNormRelu(32, 32, kernel_size=3, padding=1),
        )
        self.conv1a = ConvNormRelu(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = ConvNormRelu(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = ConvNormRelu(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = ConvNormRelu(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = ConvNormRelu(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = ConvNormRelu(160, 192, kernel_size=3, stride=2, padding=1)

        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)

        self.deconv6b = Conv2x(192, 160, deconv=True)
        self.outconv_6 = ConvNormRelu(160, 32, kernel_size=3, padding=1)

        self.deconv5b = Conv2x(160, 128, deconv=True)
        self.outconv_5 = ConvNormRelu(128, 32, kernel_size=3, padding=1)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.outconv_4 = ConvNormRelu(96, 32, kernel_size=3, padding=1)

        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.outconv_3 = ConvNormRelu(64, 32, kernel_size=3, padding=1)

        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.outconv_2 = ConvNormRelu(48, out_channels, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
        }

    def forward(self, x):

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x

        x = self.deconv6a(x, rem5)
        rem5 = x
        x = self.deconv5a(x, rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)

        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)
        x = self.deconv5b(x, rem4)
        x5 = self.outconv_5(x)
        x = self.deconv4b(x, rem3)
        x4 = self.outconv_4(x)
        x = self.deconv3b(x, rem2)
        x3 = self.outconv_3(x)
        x = self.deconv2b(x, rem1)
        x2 = self.outconv_2(x)

        return [x, x2, x3, x4, x5, x6]
