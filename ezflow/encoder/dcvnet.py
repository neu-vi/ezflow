import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY
from .residual import BasicEncoder


@ENCODER_REGISTRY.register()
class DCVNetBackbone(nn.Module):
    """
    ResNet-style encoder that outputs feature maps of size (H/2,W/2) and (H/8,W/8)
    used in  `DCVNet: Dilated Cost Volume Networks for Fast Optical Flow <https://jianghz.me/files/DCVNet_camera_ready_wacv2023.pdf>`_

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    norm : str
        Normalization layer to use. One of "batch", "instance", "group", or None
    p_dropout : float
        Dropout probability
    layer_config : list of int or tuple of int
        Number of output features per layer

    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        norm="instance",
        p_dropout=0.0,
        layer_config=(64, 96, 128),
    ):
        super(DCVNetBackbone, self).__init__()
        assert len(layer_config) == 3, "Invalid number of layers for DCVNetBackbone."

        self.encoder = BasicEncoder(
            in_channels=in_channels,
            norm=norm,
            p_dropout=p_dropout,
            layer_config=layer_config,
            num_residual_layers=(1, 2, 2),
            intermediate_features=True,
        )

        self.conv_stride2 = nn.Conv2d(layer_config[0], out_channels // 2, kernel_size=1)
        self.conv_stride8 = nn.Conv2d(layer_config[2], out_channels, kernel_size=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "norm": cfg.NORM,
            "p_dropout": cfg.P_DROPOUT,
            "layer_config": cfg.LAYER_CONFIG,
        }

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = torch.cat(x, dim=0)

        feature_pyramid = self.encoder(x)

        # Use feature maps of downsampling size (H/2,W/2) and (H/8, W/8)
        context = [feature_pyramid[0], feature_pyramid[2]]

        feat = []
        for x_i, conv_i in zip(context, [self.conv_stride2, self.conv_stride8]):
            feat.append(conv_i(x_i))

        return feat, context
