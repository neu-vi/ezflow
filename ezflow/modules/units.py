import torch.nn as nn


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
