import torch
from torch import nn


class MockOpticalFlowModel(nn.Module):
    def __init__(self, img_channels):
        super().__init__()

        self.model = nn.Conv2d(img_channels * 2, 2, kernel_size=1)

    def forward(self, img1, img2):

        x = torch.cat([img1, img2], dim=-3)

        return self.model(x)
