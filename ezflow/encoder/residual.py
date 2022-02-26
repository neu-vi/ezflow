import torch
import torch.nn as nn

from ..config import configurable
from ..modules import BasicBlock, BottleneckBlock
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class BasicEncoder(nn.Module):
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
        Configuration of encoder's layers
    intermediate_features : bool
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        norm="batch",
        p_dropout=0.0,
        layer_config=(64, 96, 128),
        intermediate_features=False,
    ):
        super(BasicEncoder, self).__init__()

        self.intermediate_features = intermediate_features

        norm = norm.lower()
        assert norm in ("group", "batch", "instance", "none")

        start_channels = layer_config[0]

        if norm == "group":
            norm_fn = nn.GroupNorm(num_groups=8, num_channels=start_channels)

        elif norm == "batch":
            norm_fn = nn.BatchNorm2d(start_channels)

        elif norm == "instance":
            norm_fn = nn.InstanceNorm2d(start_channels)

        elif norm == "none":
            norm_fn = nn.Identity()

        layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, start_channels, kernel_size=7, stride=2, padding=3
                ),
                norm_fn,
                nn.ReLU(inplace=True),
            ]
        )

        for i in range(len(layer_config)):

            if i == 0:
                stride = 1
            else:
                stride = 2

            layers.append(
                self._make_layer(start_channels, layer_config[i], stride, norm)
            )
            start_channels = layer_config[i]

        layers.append(nn.Conv2d(layer_config[-1], out_channels, kernel_size=1))

        dropout = nn.Identity()
        if self.training and p_dropout > 0:
            dropout = nn.Dropout2d(p=p_dropout)
        layers.append(dropout)

        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm):

        layer1 = BasicBlock(in_channels, out_channels, stride, norm)
        layer2 = BasicBlock(out_channels, out_channels, stride=1, norm=norm)

        return nn.Sequential(*[layer1, layer2])

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "norm": cfg.NORM,
            "p_dropout": cfg.P_DROPOUT,
            "layer_config": cfg.LAYER_CONFIG,
            "intermediate_features": cfg.INTERMEDIATE_FEATURES,
        }

    def forward(self, x):

        if self.intermediate_features is True:

            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)

                if isinstance(self.encoder[i], nn.Sequential):
                    features.append(x)

            features.append(x)

            return features

        else:

            is_list = isinstance(x, tuple) or isinstance(x, list)
            if is_list:
                batch_dim = x[0].shape[0]
                x = torch.cat(x, dim=0)

            out = self.encoder(x)

            if is_list:
                out = torch.split(out, [batch_dim, batch_dim], dim=0)

            return out


@ENCODER_REGISTRY.register()
class BottleneckEncoder(nn.Module):
    """
    ResNet-style encoder with bottleneck residual blocks

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
        Configuration of encoder's layers
    intermediate_features : bool
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        norm="batch",
        p_dropout=0.0,
        layer_config=(32, 64, 96),
        intermediate_features=False,
    ):
        super(BottleneckEncoder, self).__init__()

        self.intermediate_features = intermediate_features

        norm = norm.lower()
        assert norm in ("group", "batch", "instance", "none")

        start_channels = layer_config[0]

        if norm == "group":
            norm_fn = nn.GroupNorm(num_groups=8, num_channels=start_channels)

        elif norm == "batch":
            norm_fn = nn.BatchNorm2d(start_channels)

        elif norm == "instance":
            norm_fn = nn.InstanceNorm2d(start_channels)

        elif norm == "none":
            norm_fn = nn.Identity()

        layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, start_channels, kernel_size=7, stride=2, padding=3
                ),
                norm_fn,
                nn.ReLU(inplace=True),
            ]
        )

        for i in range(len(layer_config)):

            if i == 0:
                stride = 1
            else:
                stride = 2

            layers.append(
                self._make_layer(start_channels, layer_config[i], stride, norm)
            )
            start_channels = layer_config[i]

        layers.append(nn.Conv2d(layer_config[-1], out_channels, kernel_size=1))

        dropout = nn.Identity()
        if self.training and p_dropout > 0:
            dropout = nn.Dropout2d(p=p_dropout)
        layers.append(dropout)

        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm):

        layer1 = BottleneckBlock(in_channels, out_channels, stride, norm)
        layer2 = BottleneckBlock(out_channels, out_channels, stride=1, norm=norm)

        return nn.Sequential(*[layer1, layer2])

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "norm": cfg.NORM,
            "p_dropout": cfg.P_DROPOUT,
            "layer_config": cfg.LAYER_CONFIG,
            "intermediate_features": cfg.INTERMEDIATE_FEATURES,
        }

    def forward(self, x):

        if self.intermediate_features is True:

            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)

                if isinstance(self.encoder[i], nn.Sequential):
                    features.append(x)

            features.append(x)

            return features

        else:

            is_list = isinstance(x, tuple) or isinstance(x, list)
            if is_list:
                batch_dim = x[0].shape[0]
                x = torch.cat(x, dim=0)

            out = self.encoder(x)

            if is_list:
                out = torch.split(out, [batch_dim, batch_dim], dim=0)

            return out
