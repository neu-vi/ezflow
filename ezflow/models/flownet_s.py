import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, kaiming_normal_

from ..decoder import build_decoder
from ..encoder import build_encoder
from ..modules import BaseModule
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FlowNetS(BaseModule):
    """
    Implementation of **FlowNetSimple** from the paper
    `FlowNet: Learning Optical Flow with Convolutional Networks <https://arxiv.org/abs/1504.06852>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(FlowNetS, self).__init__()

        self.cfg = cfg

        self.encoder = build_encoder(cfg.ENCODER)

        self.decoder = build_decoder(cfg.DECODER)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

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

        H, W = img1.shape[-2:]

        x = torch.cat([img1, img2], axis=1)

        conv_outputs = self.encoder(x)

        flow_preds = self.decoder(conv_outputs)
        flow_preds.reverse()

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
