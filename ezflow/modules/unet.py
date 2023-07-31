import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import configurable
from .base_module import BaseModule
from .build import MODULE_REGISTRY, build_module


def _get_norm_fn(in_dim, norm="instance"):
    assert norm in ["instance", "batch", "none"]
    if norm == "instance":
        return nn.InstanceNorm2d(in_dim)
    elif norm == "batch":
        return nn.BatchNorm2d(in_dim)
    elif norm == "none":
        return nn.Identity()


@MODULE_REGISTRY.register()
class ASPPConv2D(nn.Module):
    """
    Applies a 2D Atrous Spatial Pyramid Pooling(ASPP) Convolution over an input image.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking Atrous Convolution for Semantic Image Segmentation."
        https://arxiv.org/pdf/1706.05587.pdf

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature
    hidden_dim : int, optional
        Number of hidden dimension
    out_channels : int, optional
        Number of output channels produced by the ASPP module
    dilations: List of int, optional
        Spacing between convolution kernels
    groups: int, optional
        Number of blocked connections from intput features to output features
    norm : str
        Type of normalization to use. Can be None, 'batch', 'group', 'instance'

    """

    @configurable
    def __init__(
        self,
        in_channels,
        hidden_dim=512,
        out_channels=512,
        dilations=(4, 8, 16),
        groups=1,
        norm="none",
    ):
        super(ASPPConv2D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 5,
                out_channels,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "hidden_dim": cfg.HIDDEN_DIMS,
            "out_channels": cfg.OUT_CHANNELS,
            "dilations": cfg.DILATIONS,
            "groups": cfg.GROUPS,
            "norm": cfg.NORM,
        }

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=False
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        out = self.bottleneck(torch.cat((feat1, feat2, feat3, feat4, feat5), 1))
        return out


@MODULE_REGISTRY.register()
class UNetBase(BaseModule):
    @configurable
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        bottle_neck_cfg,
        groups=1,
        norm="none",
    ):
        super(UNetBase, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(),
        )

        # in 1/2, out: 1/4
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=2,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
        )

        # in: 1/4, out: 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=2,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
        )

        # in: 1/8, out : 1/8 bottleneck module
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
            build_module(bottle_neck_cfg),
        )

        # in: 1/8, out: 1/4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2 + hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim * 2, norm),
            nn.ReLU(),
        )

        # in: 1/4, out: 1/2
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                hidden_dim * 2 + hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(out_channels, norm),
            nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "hidden_dim": cfg.HIDDEN_DIMS,
            "out_channels": cfg.OUT_CHANNELS,
            "bottle_neck_cfg": cfg.BOTTLE_NECK,
            "groups": cfg.GROUPS,
            "norm": cfg.NORM,
        }

    def forward(self, x):
        x = self.stem(x)

        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x3 = F.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x3 = torch.cat((x3, x1), dim=1)
        x4 = self.conv4(x3)

        x4 = F.interpolate(x4, size=x.shape[2:], mode="bilinear", align_corners=False)
        x4 = torch.cat((x4, x), dim=1)
        x5 = self.conv5(x4)
        return x5


@MODULE_REGISTRY.register()
class UNetLight(UNetBase):
    @configurable
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        bottle_neck_cfg,
        groups=1,
        norm="none",
    ):
        super(UNetLight, self).__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            bottle_neck_cfg=bottle_neck_cfg,
            groups=groups,
            norm=norm,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=groups,
            ),
            _get_norm_fn(hidden_dim, norm),
            nn.ReLU(),
        )
