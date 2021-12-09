import torch
import torch.nn as nn


class Conv2x(nn.Module):
    """
    Double convolutional layer with the option to perform deconvolution and concatenation

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    deconv : bool
        Whether to perform deconvolution instead of convolution
    concat : bool
        Whether to concatenate the input and the output of the first convolution layer
    norm : str
        Type of normalization to use. Can be "batch", "instance", "group", or "none"
    activation : str
        Type of activation to use. Can be "relu" or "leakyrelu"
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        concat=True,
        norm="batch",
        activation="relu",
    ):
        super(Conv2x, self).__init__()

        self.concat = concat
        self.deconv = deconv

        if deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = ConvNormRelu(
            in_channels,
            out_channels,
            deconv,
            kernel_size=kernel,
            stride=2,
            padding=1,
        )

        if self.concat:
            self.conv2 = ConvNormRelu(
                out_channels * 2,
                out_channels,
                deconv=False,
                norm=norm,
                activation=activation,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        else:
            self.conv2 = ConvNormRelu(
                out_channels,
                out_channels,
                deconv=False,
                norm=norm,
                activation=activation,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):

        x = self.conv1(x)

        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem

        x = self.conv2(x)

        return x


class ConvNormRelu(nn.Module):
    """
    Block for a convolutional layer with normalization and activation

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    deconv : bool, optional
        If True, use a transposed convolution
    norm : str, optional
        Normalization method
    activation : str, optional
        Activation function
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        norm="batch",
        activation="relu",
        **kwargs
    ):
        super(ConvNormRelu, self).__init__()

        bias = False

        if norm is not None:

            if norm.lower() == "group":
                self.norm = nn.GroupNorm(out_channels)

            elif norm.lower() == "batch":
                self.norm = nn.BatchNorm2d(out_channels)

            elif norm.lower() == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)

        else:
            self.norm = nn.Identity()
            bias = True

        if activation is not None:
            if activation.lower() == "leakyrelu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()

        if "kernel_size" not in kwargs.keys():
            kwargs["kernel_size"] = 3

        if deconv:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, bias=bias, **kwargs
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


def conv(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=None
):
    """
    2D convolution layer with optional normalization followed by
    an inplace LeakyReLU activation of 0.1 negative slope.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, default: 3
        Size of the convolutional kernel
    stride : int, default: 1
        Stride of the convolutional kernel
    padding : int, default: 1
        Padding of the convolutional kernel
    dilation : int, default: 1
        Dilation of the convolutional kernel
    norm : str, default: None
        Type of normalization to use. Can be None, 'batch', 'layer', 'group'
    """
    bias = False
    if norm == "group":
        norm_fn = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    elif norm == "batch":
        norm_fn = nn.BatchNorm2d(out_channels)

    elif norm == "instance":
        norm_fn = nn.InstanceNorm2d(out_channels)

    else:
        norm_fn = nn.Identity()
        bias = True

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        ),
        norm_fn,
        nn.LeakyReLU(0.1, inplace=True),
    )


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    """
    2D transpose convolution layer followed by an activation function

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, optional
        Size of the convolutional kernel
    stride : int, optional
        Stride of the convolutional kernel
    padding : int, optional
        Padding of the convolutional kernel
    dilation : int, optional
        Dilation of the convolutional kernel
    """

    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=True
    )
