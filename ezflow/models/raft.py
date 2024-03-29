import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import build_decoder
from ..encoder import build_encoder
from ..modules import BaseModule
from ..similarity import build_similarity
from ..utils import convex_upsample_flow, coords_grid, upflow
from .build import MODEL_REGISTRY

try:
    autocast = torch.cuda.amp.autocast
except:

    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


@MODEL_REGISTRY.register()
class RAFT(BaseModule):
    """
    Implementation of the paper
    `RAFT: Recurrent All-Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(RAFT, self).__init__()

        self.cfg = cfg

        self.fnet = build_encoder(cfg.ENCODER.FEATURE)
        self.cnet = build_encoder(
            cfg.ENCODER.CONTEXT, out_channels=cfg.HIDDEN_DIM + cfg.CONTEXT_DIM
        )

        self.similarity_fn = build_similarity(cfg.SIMILARITY, instantiate=False)
        self.corr_radius = cfg.CORR_RADIUS
        self.corr_levels = cfg.CORR_LEVELS

        self.update_block = build_decoder(
            name=cfg.DECODER.NAME,
            corr_radius=self.corr_radius,
            corr_levels=self.corr_levels,
            hidden_dim=cfg.HIDDEN_DIM,
            input_dim=cfg.DECODER.INPUT_DIM,
        )

    def _initialize_flow(self, img):

        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coords0, coords1

    def forward(self, img1, img2, flow_init=None):
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

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap1, fmap2 = self.fnet([img1, img2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = self.similarity_fn(
            fmap1, fmap2, num_levels=self.corr_levels, corr_radius=self.corr_radius
        )

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            cnet = self.cnet(img1)
            net, inp = torch.split(
                cnet, [self.cfg.HIDDEN_DIM, self.cfg.CONTEXT_DIM], dim=1
            )
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self._initialize_flow(img1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(self.cfg.UPDATE_ITERS):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow
            flow = coords1 - coords0
            if up_mask is None:
                flow_up = upflow(flow)
            else:
                flow_up = convex_upsample_flow(8 * flow, up_mask, out_stride=8)

            flow_predictions.append(flow_up)

        output = {"flow_preds": flow_predictions}

        if self.training:
            return output

        output["flow_upsampled"] = flow_up
        return output
