import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import configurable
from ..modules import ConvNormRelu
from .build import ENCODER_REGISTRY


class ResidualBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        dilation=1,
        norm=True,
    ):
        super(ResidualBlock, self).__init__()

        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        self.downsample = nn.Identity() if downsample is None else downsample

        if norm:
            norm = "batch"
        else:
            norm = None

        self.block = nn.Sequential(
            ConvNormRelu(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                norm=norm,
            ),
            ConvNormRelu(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activation=None,
            ),
        )

    def forward(self, x):

        residual = x
        out = self.block(x)
        residual = self.downsample(x)
        out += residual
        out = F.leaky_relu(out, 0.1)

        return out


class PyramidPooling(nn.Module):
    """
    Pyramid pooling module for the **PSPNet** feature extractor

    Parameters
    ----------
    in_channels : int
        Number of input channels
    levels : int
        Number of levels in the pyramid
    norm : bool
        Whether to use batch normalization
    """

    def __init__(self, in_channels, levels=4, norm=True):
        super(PyramidPooling, self).__init__()

        self.levels = levels
        if norm:
            norm = "batch"
        else:
            norm = None

        self.paths = []
        for _ in range(levels):
            self.paths.append(
                ConvNormRelu(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    norm=norm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):

        H, W = x.shape[2:]

        k_sizes = []
        strides = []

        for pool_size in torch.linspace(1, min(H, W) // 2, self.levels):
            k_sizes.append((int(H / pool_size), int(W / pool_size)))
            strides.append((int(H / pool_size), int(W / pool_size)))

        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):

            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
            pp_sum = pp_sum + 1.0 / self.levels * out

        pp_sum = self.relu(pp_sum / 2.0)

        return pp_sum


@ENCODER_REGISTRY.register()
class PSPNetBackbone(nn.Module):
    """
    PSPNet feature extractor backbone (https://arxiv.org/abs/1612.01105)
    Used in **Volumetric Correspondence Networks for Optical Flow** (https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)

    Parameters
    ----------
    is_proj : bool
        Whether to use projection pooling or not
    groups : int
        Number of groups in the convolutional
    in_channels : int
        Number of input channels
    norm : bool
        Whether to use batch normalization

    """

    @configurable
    def __init__(self, is_proj=True, groups=1, in_channels=3, norm=True):
        super(PSPNetBackbone, self).__init__()

        self.is_proj = is_proj
        self.inplanes = 32

        if norm:
            norm = "batch"
        else:
            norm = None

        self.convbnrelu1_1 = ConvNormRelu(
            in_channels, 16, kernel_size=3, padding=1, stride=2, norm=norm
        )
        self.convbnrelu1_2 = ConvNormRelu(
            16, 16, kernel_size=3, padding=1, stride=1, norm=norm
        )
        self.convbnrelu1_3 = ConvNormRelu(
            16, 32, kernel_size=3, padding=1, stride=1, norm=norm
        )
        self.res_block3 = self._make_layer(ResidualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.pyramid_pooling = PyramidPooling(128, levels=3, norm=norm)

        self.upconv6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm),
        )
        self.iconv5 = ConvNormRelu(
            192, 128, kernel_size=3, padding=1, stride=1, norm=norm
        )
        self.upconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm),
        )
        self.iconv4 = ConvNormRelu(
            192, 128, kernel_size=3, padding=1, stride=1, norm=norm
        )
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm),
        )
        self.iconv3 = ConvNormRelu(
            128, 64, kernel_size=3, padding=1, stride=1, norm=norm
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormRelu(64, 32, kernel_size=3, padding=1, stride=1, norm=norm),
        )
        self.iconv2 = ConvNormRelu(
            64, 64, kernel_size=3, padding=1, stride=1, norm=norm
        )

        if self.is_proj:
            self.proj6 = ConvNormRelu(
                128, 128 // groups, kernel_size=1, padding=0, stride=1
            )
            self.proj5 = ConvNormRelu(
                128, 128 // groups, kernel_size=1, padding=0, stride=1
            )
            self.proj4 = ConvNormRelu(
                128, 128 // groups, kernel_size=1, padding=0, stride=1
            )
            self.proj3 = ConvNormRelu(
                64, 64 // groups, kernel_size=1, padding=0, stride=1
            )
            self.proj2 = ConvNormRelu(
                64, 64 // groups, kernel_size=1, padding=0, stride=1
            )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg):
        return {
            "is_proj": cfg.IS_PROJ,
            "groups": cfg.GROUPS,
            "in_channels": cfg.IN_CHANNELS,
            "norm": cfg.NORM,
        }

    def forward(self, x):

        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.interpolate(
            conv6,
            [conv5.size()[2], conv5.size()[3]],
            mode="bilinear",
            align_corners=True,
        )
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)

        conv5x = F.interpolate(
            conv5,
            [conv4.size()[2], conv4.size()[3]],
            mode="bilinear",
            align_corners=True,
        )
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)

        conv4x = F.interpolate(
            conv4,
            [rconv3.size()[2], rconv3.size()[3]],
            mode="bilinear",
            align_corners=True,
        )
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)

        conv3x = F.interpolate(
            conv3,
            [pool1.size()[2], pool1.size()[3]],
            mode="bilinear",
            align_corners=True,
        )
        concat2 = torch.cat((pool1, self.upconv3[1](conv3x)), dim=1)
        conv2 = self.iconv2(concat2)

        if self.is_proj:

            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)

            return [proj6, proj5, proj4, proj3, proj2]

        return [conv6, conv5, conv4, conv3, conv2]
