import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowEntropy(nn.Module):
    """Computes entropy from matching cost"""

    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):

        x = torch.squeeze(x, 1)
        B, U, V, H, W = x.shape
        x = x.view(B, -1, H, W)
        x = F.softmax(x, dim=1).view(B, U, V, H, W)

        global_entropy = (
            (-x * torch.clamp(x, 1e-9, 1 - 1e-9).log()).sum(1).sum(1)[:, np.newaxis]
        )
        global_entropy /= np.log(x.shape[1] * x.shape[2])

        return global_entropy
