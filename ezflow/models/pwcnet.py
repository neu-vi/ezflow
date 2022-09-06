import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import ContextNetwork, build_decoder
from ..encoder import build_encoder
from ..modules import BaseModule
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PWCNet(BaseModule):
    """
    Implementation of the paper
    `PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume <https://arxiv.org/abs/1709.02371>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(PWCNet, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)

        self.decoder = build_decoder(cfg.DECODER)

        search_range = (2 * cfg.DECODER.SIMILARITY.MAX_DISPLACEMENT + 1) ** 2
        self.context_net = ContextNetwork(
            in_channels=search_range
            + cfg.DECODER.SIMILARITY.MAX_DISPLACEMENT
            + cfg.DECODER.CONFIG[-1]
            + sum(cfg.DECODER.CONFIG),
            config=cfg.DECODER.CONFIG,
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

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

        # normalize
        img1 = 1.0 * (img1 / 255.0)
        img2 = 1.0 * (img2 / 255.0)

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        flow_preds, features = self.decoder(feature_pyramid1, feature_pyramid2)

        flow_preds[-1] += self.context_net(features)

        output = {"flow_preds": flow_preds}

        if self.training:
            return output

        flow_up = flow_preds[-1]

        flow_up = F.interpolate(
            flow_up, size=(H, W), mode="bilinear", align_corners=False
        )

        flow_up *= self.cfg.DECODER.FLOW_SCALE_FACTOR

        output["flow_upsampled"] = flow_up

        return output
