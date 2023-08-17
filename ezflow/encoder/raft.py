import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY
from .residual import BasicEncoder


@ENCODER_REGISTRY.register()
class RAFTBackbone(nn.Module):
    """
    ResNet-style encoder with basic residual blocks

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
        super(RAFTBackbone, self).__init__()

        self.encoder = BasicEncoder(
            in_channels=in_channels,
            norm=norm,
            p_dropout=p_dropout,
            layer_config=layer_config,
            num_residual_layers=(2, 2, 2),
            intermediate_features=False,
        )

        self.conv_out = nn.Conv2d(layer_config[-1], out_channels, kernel_size=1)
        self.dropout = nn.Identity()
        if p_dropout > 0:
            self.dropout = nn.Dropout2d(p=p_dropout)

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
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        out = self.encoder(x)
        out = self.conv_out(out)
        out = self.dropout(out)

        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)

        return out
