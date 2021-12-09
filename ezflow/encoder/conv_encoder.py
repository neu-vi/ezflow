import torch
import torch.nn as nn

from ..config import configurable
from .build import ENCODER_REGISTRY


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, norm=None):
    """
    Generic convolutional layer with optional batch normalization

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolutional kernel
    stride : int
        Stride of the convolutional kernel
    norm : str
        Type of normalization to use. Can be None, 'batch', 'layer', 'instance'
    """

    if norm == "group":
        norm_fn = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    elif norm == "batch":
        norm_fn = nn.BatchNorm2d(out_channels)

    elif norm == "instance":
        norm_fn = nn.InstanceNorm2d(out_channels)

    else:
        norm_fn = nn.Identity()

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        ),
        norm_fn,
        nn.LeakyReLU(0.1, inplace=True),
    )


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
        Type of normalization to use. Can be None, 'batch', 'layer', 'instance'
    """

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

            self.encoder.append(
                conv_block(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=stride,
                    norm=norm,
                )
            )

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
        Type of normalization to use. Can be None, 'batch', 'layer', 'instance'
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
            conv_block(channels[0], channels[1], kernel_size=7, stride=2)
        )
        self.encoder.append(
            conv_block(channels[1], channels[2], kernel_size=5, stride=2)
        )
        self.encoder.append(
            conv_block(channels[2], channels[3], kernel_size=5, stride=2)
        )

        channels = channels[3:]

        for i in range(len(channels) - 1):

            stride = 1 if i % 2 == 0 else 2

            self.encoder.append(
                conv_block(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=stride,
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
