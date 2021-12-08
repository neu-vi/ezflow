import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class FlowNetC(nn.Module):
    """
    Implementation of FlowNetCorrelation from the paper
    **FlowNet: Learning Optical Flow with Convolutional Networks** (https://arxiv.org/abs/1504.06852)

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(FlowNetC, self).__init__()

        self.cfg = cfg


    def forward(self, img1, img2):
        """
        Performs forward pass of the network

        Parameters
        ----------
        img1 : torch.Tensor
            Image to predict flow from
        img2 : torch.Tensor
            Image to predict flow to

        Returns
        -------
        torch.Tensor
            Flow from img1 to img2
        """

        pass