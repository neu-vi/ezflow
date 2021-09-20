import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import RecurrentLookupUpdateBlock, SmallRecurrentLookupUpdateBlock
from ...encoder import BasicEncoder, BottleneckEncoder
from ...similarity import MutliScalePairwise4DCorr
from ...utils import coords_grid, upflow
from ..model_zoo import MODEL_REGISTRY

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
class RAFT(nn.Module):
    def __init__(
        self,
        small=False,
        dropout=0.0,
        mixed_precision=False,
    ):
        super(RAFT, self).__init__()

        self.mixed_precision = mixed_precision

        if small:

            self.hidden_dim = 96
            self.context_dim = 64
            self.corr_radius = 3
            self.corr_levels = 4

            self.fnet = BottleneckEncoder(
                out_channels=128, norm="instance", p_dropout=dropout
            )
            self.cnet = BottleneckEncoder(
                out_channels=self.hidden_dim + self.context_dim,
                norm="none",
                p_dropout=dropout,
            )
            self.update_block = SmallRecurrentLookupUpdateBlock(
                corr_radius=self.corr_radius,
                corr_levels=self.corr_levels,
                hidden_dim=self.hidden_dim,
            )

        else:

            self.hidden_dim = 128
            self.context_dim = 128
            self.corr_levels = 4
            self.corr_radius = 4

            self.fnet = BasicEncoder(
                out_channels=256, norm="instance", p_dropout=dropout
            )
            self.cnet = BasicEncoder(
                out_channels=self.hidden_dim + self.context_dim,
                norm="batch",
                p_dropout=dropout,
            )
            self.update_block = RecurrentLookupUpdateBlock(
                corr_radius=self.corr_radius,
                corr_levels=self.corr_levels,
                hidden_dim=self.hidden_dim,
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):

        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self,
        image1,
        image2,
        iters=12,
        flow_init=None,
        upsample=True,
        only_flow=True,
        test_mode=False,
    ):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = MutliScalePairwise4DCorr(fmap1, fmap2, radius=self.corr_radius)

        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode or not self.training:
            if only_flow:
                return flow_up
            return coords1 - coords0, flow_up

        return flow_predictions
