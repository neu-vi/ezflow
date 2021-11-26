import torch
import torch.nn as nn

from ...config import configurable
from ..build import MODULE_REGISTRY


@MODULE_REGISTRY.register()
class ConvGRU(nn.Module):
    """
    Convolutinal GRU layer

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension of the GRU
    input_dim : int, optional
        Input dimension of the GRU
    kernel_size : int, optional
        Kernel size of the convolutional layers
    """

    @configurable
    def __init__(self, hidden_dim=128, input_dim=192 + 128, kernel_size=3):
        super(ConvGRU, self).__init__()

        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=1
        )
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=1
        )
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=1
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.HIDDEN_DIM,
            "input_dim": cfg.INPUT_DIM,
            "kernel_size": cfg.KERNEL_SIZE,
        }

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q

        return h
