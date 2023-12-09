import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import build_decoder
from ..encoder import build_encoder
from ..modules import BaseModule, build_module
from ..similarity import build_similarity
from ..utils import replace_relu
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

        self.encoder = build_encoder(self.cfg.ENCODER)
        self.cost_volume_list = build_similarity(self.cfg.SIMILARITY)

        if "DILATIONS" not in self.cfg.DECODER:
            self.cfg.DECODER.DILATIONS = self.cfg.SIMILARITY.DILATIONS

        if "SEARCH_RANGE" not in self.cfg.DECODER.COST_VOLUME_FILTER:
            self.cfg.DECODER.COST_VOLUME_FILTER.SEARCH_RANGE = (
                self.cost_volume_list.get_search_range()
            )

        self.decoder = build_decoder(self.cfg.DECODER)
        self = replace_relu(self, nn.LeakyReLU(negative_slope=0.1))

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
            <flow_logits> torch.Tensor : interpolated flow logits
            <flow_upsampled> torch.Tensor : if model is in eval state, return upsampled flow
        """
        N, C, H, W = img1.shape
        feat_map, context_map = self.encoder([img1, img2])
        fmap1 = [feat_i[:N] for feat_i in feat_map]
        fmap2 = [feat_i[N:] for feat_i in feat_map]
        context_fmap1 = [context_i[:N] for context_i in context_map]

        assert len(fmap1) == len(self.cfg.SIMILARITY.DILATIONS)
        assert len(fmap1) == len(self.cfg.SIMILARITY.DILATIONS)

        cost = self.cost_volume_list(fmap1, fmap2)
        flow_offsets = self.cost_volume_list.get_global_flow_offsets().view(
            1, -1, 2, 1, 1
        )

        flow_list, flow_logits_list = self.decoder(cost, context_fmap1, flow_offsets)

        output = {"flow_preds": flow_list, "flow_logits": flow_logits_list}

        if self.training:
            return output

        output["flow_upsampled"] = flow_list[-1]
        return output
