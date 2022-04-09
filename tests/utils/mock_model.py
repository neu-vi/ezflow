from torch import nn


class MockOpticalFlowModel(nn.Module):
    def __init__(self, img_channels):
        super().__init__()

        self.model = nn.Conv2d(img_channels, 2, kernel_size=1)

    def forward(self, x):
        return self.model(x)
