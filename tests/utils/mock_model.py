import torch
import torch.nn.functional as F
from torch import nn

from ezflow.modules import BaseModule


class MockOpticalFlowModel(BaseModule):
    def __init__(self, img_channels):
        super().__init__()

        self.model = nn.Conv2d(img_channels * 2, 2, kernel_size=1)

    def forward(self, img1, img2):

        x = torch.cat([img1, img2], dim=-3)
        mock_flow_prediction = self.model(x)

        flow_up = F.interpolate(
            mock_flow_prediction, img1.shape[-2:], mode="bilinear", align_corners=True
        )
        output = {"flow_preds": [mock_flow_prediction], "flow_upsampled": flow_up}
        return output
