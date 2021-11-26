import torch
import torch.nn.functional as F

from ...config import configurable
from ...utils import bilinear_sampler
from ..build import SIMILARITY_REGISTRY


@SIMILARITY_REGISTRY.register()
class MutliScalePairwise4DCorr:
    """
    Pairwise 4D correlation at multiple scales. Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    fmap1 : torch.Tensor
        First feature map
    fmap2 : torch.Tensor
        Second feature map
    num_levels : int
        Number of levels in the feature pyramid
    corr_radius : int
        Radius of the correlation window
    """

    @configurable
    def __init__(self, fmap1, fmap2, num_levels=4, corr_radius=4):

        self.num_levels = num_levels
        self.corr_radius = corr_radius
        self.corr_pyramid = []

        corr = MutliScalePairwise4DCorr.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):

        r = self.corr_radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):

            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_levels": cfg.NUM_LEVELS,
            "corr_radius": cfg.CORR_RADIUS,
        }

    @staticmethod
    def corr(fmap1, fmap2):

        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        return corr / torch.sqrt(torch.tensor(dim).float())
