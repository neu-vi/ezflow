import torch
import torch.nn as nn

from ..config import configurable
from ..modules import conv
from .build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class BasicConvEncoder(nn.Module):
    """
    A Basic Convolution Encoder with a fixed size kernel = 3, padding=1 and dilation = 1.
    Every alternate layer has stride = 1 followed by stride = 2.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration for the layers in the encoder
    norm : str
        Type of normalization to use. Can be None, 'batch', 'group', 'instance'
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        config=[64, 128, 256, 512],
        norm=None,
    ):
        super(BasicConvEncoder, self).__init__()

        if isinstance(config, tuple):
            config = list(config)

        channels = [in_channels] + config

        self.encoder = nn.ModuleList()

        for i in range(len(channels) - 1):

            stride = 1 if i % 2 == 0 else 2
            kernel_size = 3

            self.encoder.append(
                conv(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    norm=norm,
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "config": cfg.CONFIG,
            "norm": cfg.NORM,
        }

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        List[torch.Tensor],
            List of all the output convolutions from each encoder layer
        """

        outputs = []

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

            if len(outputs) > 0:
                prev_output = outputs[-1]
                if prev_output.shape[1:] == x.shape[1:]:
                    outputs[-1] = x
                else:
                    outputs.append(x)
            else:
                outputs.append(x)

        return outputs


@ENCODER_REGISTRY.register()
class FlowNetConvEncoder(BasicConvEncoder):
    """
    Convolutional encoder based on the FlowNet architecture
    Used in **FlowNet: Learning Optical Flow with Convolutional Networks** (https://arxiv.org/abs/1504.06852)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration for the layers in the encoder
    norm : str
        Type of normalization to use. Can be None, 'batch', 'group', 'instance'
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        config=[64, 128, 256, 512],
        norm=None,
    ):
        super(FlowNetConvEncoder, self).__init__()

        assert (
            len(config) >= 2
        ), "FlowNetConvEncoder expects at least 2 output channels in config."

        if isinstance(config, tuple):
            config = list(config)

        channels = [in_channels] + config

        self.encoder = nn.ModuleList()

        self.encoder.append(
            conv(
                channels[0], channels[1], kernel_size=7, stride=2, padding=(7 - 1) // 2
            )
        )
        self.encoder.append(
            conv(
                channels[1], channels[2], kernel_size=5, stride=2, padding=(5 - 1) // 2
            )
        )
        self.encoder.append(
            conv(
                channels[2], channels[3], kernel_size=5, stride=2, padding=(5 - 1) // 2
            )
        )

        channels = channels[3:]

        for i in range(len(channels) - 1):

            stride = 1 if i % 2 == 0 else 2
            kernel_size = 3

            self.encoder.append(
                conv(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    norm=norm,
                )
            )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "config": cfg.CONFIG,
            "norm": cfg.NORM,
        }
