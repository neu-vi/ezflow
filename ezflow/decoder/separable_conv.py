import torch.nn.functional as F
from torch import nn

from ..config import configurable
from .build import DECODER_REGISTRY


class FeatureProjection4D(nn.Module):
    """
    Applies a 3D convolution to the input feature map

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Norm 3D
    groups : int, default : 1
        Number of groupds for 3D convolution, in_channels and out_channels must be divisible by groups
    """

    def __init__(self, in_channels, out_channels, stride, norm=True, groups=1):
        super(FeatureProjection4D, self).__init__()

        self.norm = norm
        self.stride = stride
        bias = not norm

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            1,
            (stride, stride, 1),
            padding=0,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):

        B, C, U, V, H, W = x.size()
        x = self.conv1(x.view(B, C, U, V, H * W))

        if self.norm:
            x = self.bn(x)

        _, C, U, V, _ = x.shape
        x = x.view(B, C, U, V, H, W)

        return x


@DECODER_REGISTRY.register()
class SeparableConv4D(nn.Module):
    """
    Applies two 3D convolution followed by an
    optional 2D convolution to the input feature map.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    k_size : int, default : 3
        Size of the kernel
    full : bool, default : True
        If True, applies a stride of (1, 1, 1)
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    @configurable
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=(1, 1, 1),
        norm=True,
        k_size=3,
        full=True,
        groups=1,
    ):
        super(SeparableConv4D, self).__init__()

        bias = not norm
        self.is_proj = False
        self.stride = stride[0]
        expand = 1

        if norm:
            if in_channels != out_channels:
                self.is_proj = True
                self.proj = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        1,
                        bias=bias,
                        padding=0,
                        groups=groups,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            if full:
                self.conv1 = nn.Sequential(
                    nn.Conv3d(
                        in_channels * expand,
                        in_channels,
                        (1, k_size, k_size),
                        stride=(1, self.stride, self.stride),
                        bias=bias,
                        padding=(0, k_size // 2, k_size // 2),
                        groups=groups,
                    ),
                    nn.BatchNorm3d(in_channels),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv3d(
                        in_channels * expand,
                        in_channels,
                        (1, k_size, k_size),
                        stride=1,
                        bias=bias,
                        padding=(0, k_size // 2, k_size // 2),
                        groups=groups,
                    ),
                    nn.BatchNorm3d(in_channels),
                )
            self.conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    in_channels * expand,
                    (k_size, k_size, 1),
                    stride=(self.stride, self.stride, 1),
                    bias=bias,
                    padding=(k_size // 2, k_size // 2, 0),
                    groups=groups,
                ),
                nn.BatchNorm3d(in_channels * expand),
            )
        else:
            if in_channels != out_channels:
                self.is_proj = True
                self.proj = nn.Conv2d(
                    in_channels, out_channels, 1, bias=bias, padding=0, groups=groups
                )
            if full:
                self.conv1 = nn.Conv3d(
                    in_channels * expand,
                    in_channels,
                    (1, k_size, k_size),
                    stride=(1, self.stride, self.stride),
                    bias=bias,
                    padding=(0, k_size // 2, k_size // 2),
                    groups=groups,
                )
            else:
                self.conv1 = nn.Conv3d(
                    in_channels * expand,
                    in_channels,
                    (1, k_size, k_size),
                    stride=1,
                    bias=bias,
                    padding=(0, k_size // 2, k_size // 2),
                    groups=groups,
                )
            self.conv2 = nn.Conv3d(
                in_channels,
                in_channels * expand,
                (k_size, k_size, 1),
                stride=(self.stride, self.stride, 1),
                bias=bias,
                padding=(k_size // 2, k_size // 2, 0),
                groups=groups,
            )
        self.relu = nn.ReLU(inplace=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "stride": cfg.STRIDE,
            "norm": cfg.NORM,
            "k_size": cfg.k_size,
            "full": cfg.FULL,
            "groups": cfg.GROUPS,
        }

    def forward(self, x):

        B, C, U, V, H, W = x.shape

        x = self.conv2(x.view(B, C, U, V, -1))
        B, C, U, V, _ = x.shape

        x = self.relu(x)
        x = self.conv1(x.view(B, C, -1, H, W))

        B, C, _, H, W = x.shape
        if self.is_proj:
            x = self.proj(x.view(B, W, -1, W))

        x = x.view(B, -1, U, V, H, W)

        return x


class SeparableConv4DBlock(nn.Module):
    """
    Applies separate SeperableConv4d convolutions to the input feature map.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    k_size : int, default : 3
        Size of the kernel
    full : bool, default : True
        If True, applies SeparableConv4D otherwise FeatureProjection4D
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=(1, 1, 1),
        norm=True,
        full=True,
        groups=1,
    ):
        super(SeparableConv4DBlock, self).__init__()

        if in_channels == out_channels and stride == (1, 1, 1):
            self.downsample = None
        else:
            if full:
                self.downsample = SeparableConv4D(
                    in_channels,
                    out_channels,
                    stride,
                    norm=norm,
                    k_size=1,
                    full=full,
                    groups=groups,
                )
            else:
                self.downsample = FeatureProjection4D(
                    in_channels, out_channels, stride[0], norm=norm, groups=groups
                )
        self.conv1 = SeparableConv4D(
            in_channels, out_channels, stride, norm=norm, full=full, groups=groups
        )
        self.conv2 = SeparableConv4D(
            out_channels, out_channels, (1, 1, 1), norm=norm, full=full, groups=groups
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu1(self.conv1(x))

        if self.downsample:
            x = self.downsample(x)

        out = self.relu2(x + self.conv2(out))

        return out


@DECODER_REGISTRY.register()
class Butterfly4D(nn.Module):
    """
    Applies a FeatureProjection4D followed by
    five SeperableConv4d convolutions to the input feature map.

    Parameters
    ----------
    f_dim_1 : int
        Number of input channels
    f_dim_2 : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    full : bool, default : True
        If True, applies SeparableConv4D otherwise FeatureProjection4D
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    @configurable
    def __init__(self, f_dim_1, f_dim_2, norm=True, full=True, groups=1):
        super(Butterfly4D, self).__init__()

        self.proj = nn.Sequential(
            FeatureProjection4D(f_dim_1, f_dim_2, 1, norm=norm, groups=groups),
            nn.ReLU(inplace=True),
        )
        self.conva1 = SeparableConv4DBlock(
            f_dim_2, f_dim_2, norm=norm, stride=(2, 1, 1), full=full, groups=groups
        )
        self.conva2 = SeparableConv4DBlock(
            f_dim_2, f_dim_2, norm=norm, stride=(2, 1, 1), full=full, groups=groups
        )
        self.convb3 = SeparableConv4DBlock(
            f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups
        )
        self.convb2 = SeparableConv4DBlock(
            f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups
        )
        self.convb1 = SeparableConv4DBlock(
            f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "f_dim_1": cfg.F_DIM_1,
            "f_dim_2": cfg.F_DIM_2,
            "norm": cfg.NORM,
            "full": cfg.FULL,
            "groups": cfg.GROUPS,
        }

    def forward(self, x):

        out = self.proj(x)
        B, C, U, V, H, W = out.shape

        out1 = self.conva1(out)
        _, _, U1, V1, H1, W1 = out1.shape

        out2 = self.conva2(out1)
        _, _, U2, V2, H2, W2 = out2.shape

        out2 = self.convb3(out2)

        t_out_1 = F.interpolate(
            out2.view(B, C, U2, V2, -1),
            (U1, V1, H2 * W2),
            mode="trilinear",
            align_corners=True,
        ).view(B, C, U1, V1, H2, W2)
        t_out_1 = F.interpolate(
            t_out_1.view(B, C, -1, H2, W2),
            (U1 * V1, H1, W1),
            mode="trilinear",
            align_corners=True,
        ).view(B, C, U1, V1, H1, W1)
        out1 = t_out_1 + out1
        out1 = self.convb2(out1)

        t_out = F.interpolate(
            out1.view(B, C, U1, V1, -1),
            (U, V, H1 * W1),
            mode="trilinear",
            align_corners=True,
        ).view(B, C, U, V, H1, W1)
        t_out = F.interpolate(
            t_out.view(B, C, -1, H1, W1),
            (U * V, H, W),
            mode="trilinear",
            align_corners=True,
        ).view(B, C, U, V, H, W)

        out = t_out + out
        out = self.convb1(out)

        return out
