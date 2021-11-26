# Adapted from https://github.com/oblime/CorrelationLayer


import torch
import torch.nn as nn

from ...config import configurable
from ..build import SIMILARITY_REGISTRY


@SIMILARITY_REGISTRY.register()
class CorrelationLayer(nn.Module):
    """
    Correlation layer in pure PyTorch
    Only supports specific values of the following parameters:
    - kernel_size: 1
    - stride_1: 1
    - stride_2: 1
    - corr_multiply: 1

    Parameters
    ----------
    pad_size : int
        Padding size for the correlation layer
    max_displacement : int
        Maximum displacement for the correlation computation
    """

    @configurable
    def __init__(self, pad_size=4, max_displacement=4):
        super().__init__()

        self.max_h_disp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    @classmethod
    def from_config(cls, cfg):
        return {
            "pad_size": cfg.PAD_SIZE,
            "max_displacement": cfg.MAX_DISPLACEMENT,
        }

    def forward(self, features1, features2):

        features2_pad = self.padlayer(features2)
        offsety, offsetx = torch.meshgrid(
            [
                torch.arange(0, 2 * self.max_h_disp + 1),
                torch.arange(0, 2 * self.max_h_disp + 1),
            ]
        )

        H, W = features1.shape[2], features1.shape[3]
        output = torch.cat(
            [
                torch.mean(
                    features1 * features2_pad[:, :, dy : dy + H, dx : dx + W],
                    1,
                    keepdim=True,
                )
                for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
            ],
            1,
        )

        return output
