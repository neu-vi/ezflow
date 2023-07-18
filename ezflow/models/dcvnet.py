import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import build_decoder
from ..encoder import build_encoder
from ..modules import BaseModule
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DCVNet(BaseModule):
    """
    Implementation of **DCVNet** from the paper
    `DCVNet: Dilated Cost Volume Networks for Fast Optical Flow <https://jianghz.me/files/DCVNet_camera_ready_wacv2023.pdf>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(DCVNet, self).__init__()

        self.cfg = cfg

        self.encoder = build_encoder(cfg.ENCODER)

        # self.decoder = build_decoder(cfg.DECODER)

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
        :class:`dict`
            <flow_preds> torch.Tensor : intermediate flow predications from img1 to img2
            <flow_upsampled> torch.Tensor : if model is in eval state, return upsampled flow
        """

        flow_preds = []
        output = {"flow_preds": flow_preds}

        if self.training:
            return output

        flow = flow_preds[0]

        H_, W_ = flow.shape[-2:]
        flow = F.interpolate(
            flow, img1.shape[-2:], mode="bilinear", align_corners=False
        )
        flow_u = flow[:, 0, :, :] * (W / W_)
        flow_v = flow[:, 1, :, :] * (H / H_)
        flow = torch.stack([flow_u, flow_v], dim=1)

        output["flow_upsampled"] = flow
        return output
