import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..build import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class SoftArg2DFlowRegression(nn.Module):
    """
    Applies 2D soft argmin/argmax operation to regress flow.
    Used in **DICL** (https://arxiv.org/abs/2010.14851)

    Parameters
    ----------
    max_u : int, default : 3
        Maximum displacement in the horizontal direction
    max_v : int, default : 3
        Maximum displacement in the vertical direction
    operation : str, default : argmax
        The argmax/argmin operation for flow regression
    """

    @configurable
    def __init__(self, max_u=3, max_v=3, operation="argmax"):
        super(SoftArg2DFlowRegression, self).__init__()

        assert (
            operation.lower() == "argmax" or operation.lower() == "argmin"
        ), "Invalid operation. Supported operations: argmax and argmin"

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
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W representing the flow
        """

        sizeU = 2 * self.max_u + 1
        sizeV = 2 * self.max_v + 1
        x = x.squeeze(1)
        B, _, _, H, W = x.shape

        disp_u = torch.reshape(
            torch.arange(
                -self.max_u,
                self.max_u + 1,
                dtype=torch.float32,
            ),
            [1, sizeU, 1, 1, 1],
        ).to(x.device)
        disp_u = disp_u.expand(B, -1, sizeV, H, W).contiguous()
        disp_u = disp_u.view(B, sizeU * sizeV, H, W)

        disp_v = torch.reshape(
            torch.arange(
                -self.max_v,
                self.max_v + 1,
                dtype=torch.float32,
            ),
            [1, 1, sizeV, 1, 1],
        ).to(x.device)
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


@DECODER_REGISTRY.register()
class Soft4DFlowRegression(nn.Module):
    """
    Applies 4D soft argmax operation to regress flow.

    Parameters
    ----------
    size : List[int]
        List containing values of B, H, W
    max_disp : int, default : 4
        Maximum displacement
    entropy : bool, default : False
        If True, computes local and global entropy from matching cost
    factorization : int, default : 1
        Max displacement factorization value
    """

    @configurable
    def __init__(self, size, max_disp=4, entropy=False, factorization=1):
        super(Soft4DFlowRegression, self).__init__()

        B, H, W = size
        self.entropy = entropy
        self.md = max_disp
        self.factorization = factorization
        self.truncated = True
        self.w_size = 3

        flowrange_y = range(-max_disp, max_disp + 1)
        flowrange_x = range(
            -int(max_disp // self.factorization),
            int(max_disp // self.factorization) + 1,
        )
        meshgrid = np.meshgrid(flowrange_x, flowrange_y)
        flow_y = np.tile(
            np.reshape(
                meshgrid[0],
                [
                    1,
                    2 * max_disp + 1,
                    2 * int(max_disp // self.factorization) + 1,
                    1,
                    1,
                ],
            ),
            (B, 1, 1, H, W),
        )
        flow_x = np.tile(
            np.reshape(
                meshgrid[1],
                [
                    1,
                    2 * max_disp + 1,
                    2 * int(max_disp // self.factorization) + 1,
                    1,
                    1,
                ],
            ),
            (B, 1, 1, H, W),
        )
        self.register_buffer("flow_x", torch.Tensor(flow_x))
        self.register_buffer("flow_y", torch.Tensor(flow_y))

        self.pool3d = nn.MaxPool3d(
            (self.w_size * 2 + 1, self.w_size * 2 + 1, 1),
            stride=1,
            padding=(self.w_size, self.w_size, 0),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "size": cfg.SIZE,
            "max_disp": cfg.MAX_DISP,
            "entropy": cfg.ENTROPY,
            "factorization": cfg.FACTORIZATION,
        }

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input cost feature map of shape B x U x V x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape B x C x H x W representing the flow

        torch.Tensor
            A tensor representing the local and global entropy cost
        """
        B, U, V, H, W = x.shape
        orig_x = x

        if self.truncated:
            # truncated softmax
            x = x.view(B, U * V, H, W)

            idx = x.argmax(1)[:, np.newaxis]

            mask = torch.FloatTensor(B, U * V, H, W).fill_(0)
            mask.scatter_(1, idx, 1)
            mask = mask.view(B, 1, U, V, -1)
            mask = self.pool3d(mask)[:, 0].view(B, U, V, H, W)

            n_inf = x.clone().fill_(-np.inf).view(B, U, V, H, W)
            x = torch.where(mask.byte(), orig_x, n_inf)

        else:
            self.w_size = (np.sqrt(U * V) - 1) / 2

        B, U, V, H, W = x.shape

        x = F.softmax(x.view(B, -1, H, W), 1).view(B, U, V, H, W)
        out_x = torch.sum(torch.sum(x * self.flow_x, 1), 1, keepdim=True)
        out_y = torch.sum(torch.sum(x * self.flow_y, 1), 1, keepdim=True)

        if self.entropy:
            # local
            local_entropy = (
                (-x * torch.clamp(x, 1e-9, 1 - 1e-9).log()).sum(1).sum(1)[:, np.newaxis]
            )
            if self.w_size == 0:
                local_entropy[:] = 1.0
            else:
                local_entropy /= np.log((self.w_size * 2 + 1) ** 2)

            # global
            x = F.softmax(orig_x.view(B, -1, H, W), 1).view(B, U, V, H, W)
            global_entropy = (
                (-x * torch.clamp(x, 1e-9, 1 - 1e-9).log()).sum(1).sum(1)[:, np.newaxis]
            )
            global_entropy /= np.log(x.shape[1] * x.shape[2])

            return torch.cat([out_x, out_y], 1), torch.cat(
                [local_entropy, global_entropy], 1
            )

        else:
            return torch.cat([out_x, out_y], 1), None
