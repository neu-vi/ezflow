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
        Number of output features per layer
    num_residual_layers : list of int or tuple of int
        Number of residual blocks features per layer
    intermediate_features : bool, default False
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        norm="batch",
        p_dropout=0.0,
        layer_config=(64, 96, 128),
        num_residual_layers=(2, 2, 2),
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

            stride = 1 if i == 0 else 2

            layers.append(
                self._make_layer(
                    start_channels,
                    layer_config[i],
                    stride,
                    norm,
                    num_residual_layers[i],
                )
            )
            start_channels = layer_config[i]

        self.dropout = nn.Identity()
        if p_dropout > 0:
            self.dropout = nn.Dropout2d(p=p_dropout)

        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm, num_layers=2):
        layers = [BasicBlock(in_channels, out_channels, stride, norm)]
        for _ in range(num_layers - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, norm=norm))

        return nn.Sequential(*layers)

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
            "norm": cfg.NORM,
            "p_dropout": cfg.P_DROPOUT,
            "layer_config": cfg.LAYER_CONFIG,
            "num_residual_layers": cfg.NUM_RESIDUAL_LAYERS,
            "intermediate_features": cfg.INTERMEDIATE_FEATURES,
        }

    def forward(self, x):

        if self.intermediate_features:

            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)

                if isinstance(self.encoder[i], nn.Sequential):
                    x = self.dropout(x)
                    features.append(x)

            return features

        out = self.encoder(x)
        out = self.dropout(out)
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
    intermediate_features : bool, default False
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        norm="batch",
        p_dropout=0.0,
        layer_config=(32, 64, 96),
        num_residual_layers=(2, 2, 2),
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

            stride = 1 if i == 0 else 2

            layers.append(
                self._make_layer(
                    start_channels,
                    layer_config[i],
                    stride,
                    norm,
                    num_residual_layers[i],
                )
            )
            start_channels = layer_config[i]

        self.dropout = nn.Identity()
        if p_dropout > 0:
            self.dropout = nn.Dropout2d(p=p_dropout)

        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm, num_layers=2):
        layers = [BottleneckBlock(in_channels, out_channels, stride, norm)]
        for _ in range(num_layers - 1):
            layers.append(
                BottleneckBlock(out_channels, out_channels, stride=1, norm=norm)
            )

        return nn.Sequential(*layers)

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
            "norm": cfg.NORM,
            "p_dropout": cfg.P_DROPOUT,
            "layer_config": cfg.LAYER_CONFIG,
            "num_residual_layers": cfg.NUM_RESIDUAL_LAYERS,
            "intermediate_features": cfg.INTERMEDIATE_FEATURES,
        }

    def forward(self, x):
        if self.intermediate_features:

            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)

                if isinstance(self.encoder[i], nn.Sequential):
                    x = self.dropout(x)
                    features.append(x)

            return features

        out = self.encoder(x)
        out = self.dropout(out)
        return out
