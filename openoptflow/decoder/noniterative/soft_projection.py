import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..build import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class SoftArgFlowRegression(nn.Module):
    """2D soft argmin/argmax"""

    @configurable
    def __init__(self, max_u, max_v, operation="argmax"):
        super(SoftArgFlowRegression, self).__init__()

        self.max_u = max_u
        self.max_v = max_v
        self.operation = operation.lower()

    @classmethod
    def from_config(cls, cfg):
        return {
            "max_u": cfg.MAX_U,
            "max_v": cfg.MAX_V,
            "operation": cfg.OPERATION,
        }

    def forward(self, x):

        sizeU = 2 * self.max_u + 1
        sizeV = 2 * self.max_v + 1
        x = x.squeeze(1)
        B, _, _, H, W = x.shape

        with torch.cuda.device_of(x):

            disp_u = torch.reshape(
                torch.arange(
                    -self.max_u,
                    self.max_u + 1,
                    dtype=torch.float32,
                ),
                [1, sizeU, 1, 1, 1],
            )
            disp_u = disp_u.expand(B, -1, sizeV, H, W).contiguous()
            disp_u = disp_u.view(B, sizeU * sizeV, H, W)

            disp_v = torch.reshape(
                torch.arange(
                    -self.max_v,
                    self.max_v + 1,
                    dtype=torch.float32,
                ),
                [1, 1, sizeV, 1, 1],
            )
            disp_v = disp_v.expand(B, sizeU, -1, H, W).contiguous()
            disp_v = disp_v.view(B, sizeU * sizeV, H, W)

        x = x.view(B, sizeU * sizeV, H, W)

        if self.operation == "argmin":
            x = F.softmin(x, dim=1)
        else:
            x = F.softmax(x, dim=1)

        flow_u = (x * disp_u).sum(dim=1)
        flow_v = (x * disp_v).sum(dim=1)
        flow = torch.cat((flow_u.unsqueeze(1), flow_v.unsqueeze(1)), dim=1)

        return flow
