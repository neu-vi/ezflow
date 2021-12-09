import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, kaiming_normal_

from ..decoder import build_decoder
from ..encoder import build_encoder
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FlowNetS(nn.Module):
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
        torch.Tensor
            Flow from img1 to img2
        """

        H, W = img1.shape[-2:]

        x = torch.cat([img1, img2], axis=1)

        conv_outputs = self.encoder(x)

        flow_preds = self.decoder(conv_outputs)
        flow_preds.reverse()

        if self.training:
            return flow_preds

        else:

            flow = flow_preds[0]

            if self.cfg.INTERPOLATE_FLOW:
                H_, W_ = flow.shape[-2:]
                flow = F.interpolate(
                    flow, img1.shape[-2:], mode="bilinear", align_corners=True
                )
                flow_u = flow[:, 0, :, :] * (W / W_)
                flow_v = flow[:, 1, :, :] * (H / H_)
                flow = torch.stack([flow_u, flow_v], dim=1)

            return flow
