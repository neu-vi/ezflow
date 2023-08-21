import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except:
    from .correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler

from ..config import configurable
from ..modules import ConvNormRelu
from .build import SIMILARITY_REGISTRY


@SIMILARITY_REGISTRY.register()
class Conv2DMatching(nn.Module):
    """
    Convolutional matching/filtering network for cost volume learning

    Parameters
    ----------
    config : tuple of int or list of int
        Configuration of the convolutional layers in the network
    """

    @configurable
    def __init__(self, config=(64, 96, 128, 64, 32, 1)):
        super(Conv2DMatching, self).__init__()

        self.matching_net = nn.Sequential(
            ConvNormRelu(config[0], config[1], kernel_size=3, padding=1, dilation=1),
            ConvNormRelu(config[1], config[2], kernel_size=3, stride=2, padding=1),
            ConvNormRelu(config[2], config[2], kernel_size=3, padding=1, dilation=1),
            ConvNormRelu(config[2], config[3], kernel_size=3, padding=1, dilation=1),
            ConvNormRelu(
                config[3], config[4], kernel_size=4, padding=1, stride=2, deconv=True
            ),
            nn.Conv2d(
                config[4], config[5], kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg.CONFIG,
        }

    def forward(self, x):

        x = self.matching_net(x)

        return x


@SIMILARITY_REGISTRY.register()
class Custom2DConvMatching(nn.Module):
    """
    Convolutional matching/filtering network for cost volume learning with custom convolutions

    Parameters
    ----------
    config : tuple of int or list of int
        Configuration of the convolutional layers in the network
    kernel_size : int
        Kernel size of the convolutional layers
    **kwargs
        Additional keyword arguments for the convolutional layers
    """

    @configurable
    def __init__(self, config=(16, 32, 16, 1), kernel_size=3, **kwargs):
        super(Custom2DConvMatching, self).__init__()

        matching_net = nn.ModuleList()

        for i in range(len(config) - 2):
            matching_net.append(
                ConvNormRelu(
                    config[i], config[i + 1], kernel_size=kernel_size, **kwargs
                )
            )
        matching_net.append(nn.Conv2d(config[-2], config[-1], kernel_size=1))

        self.matching_net = nn.Sequential(*matching_net)

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg.CONFIG,
            "kernel_size": cfg.KERNEL_SIZE,
        }

    def forward(self, x):

        x = self.matching_net(x)

        return x


@SIMILARITY_REGISTRY.register()
class LearnableMatchingCost(nn.Module):
    """
    Learnable matching cost network for cost volume learning. Used in **DICL** (https://arxiv.org/abs/2010.14851)

    Parameters
    ----------
    max_u : int, optional
        Maximum displacement in the horizontal direction
    max_v : int, optional
        Maximum displacement in the vertical direction
    config : tuple of int or list of int, optional
        Configuration of the convolutional layers (matching net) in the network
    remove_warp_hole : bool, optional
        Whether to remove the warp holes in the cost volume
    cuda_cost_compute : bool, optional
        Whether to compute the cost volume on the GPU
    matching_net : Optional[nn.Module], optional
        Custom matching network, by default None, which uses a Conv2DMatching network
    """

    @configurable
    def __init__(
        self,
        max_u=3,
        max_v=3,
        config=(64, 96, 128, 64, 32, 1),
        remove_warp_hole=True,
        cuda_cost_compute=False,
        matching_net=None,
    ):
        super(LearnableMatchingCost, self).__init__()

        if matching_net is not None:
            self.matching_net = matching_net
        else:
            self.matching_net = Conv2DMatching(config=config)

        self.max_u = max_u
        self.max_v = max_v
        self.remove_warp_hole = remove_warp_hole
        self.cuda_cost_compute = cuda_cost_compute

    @classmethod
    def from_config(cls, cfg):

        return {
            "max_u": cfg.MAX_U,
            "max_v": cfg.MAX_V,
            "config": cfg.CONFIG,
            "remove_warp_hole": cfg.REMOVE_WARP_HOLE,
        }

    def forward(self, x, y):

        size_u = 2 * self.max_u + 1
        size_v = 2 * self.max_v + 1
        _, c, height, width = x.shape

        with torch.cuda.device_of(x):

            cost = (
                x.new()
                .resize_(
                    x.size()[0],
                    2 * c,
                    2 * self.max_u + 1,
                    2 * self.max_v + 1,
                    height,
                    width,
                )
                .zero_()
            )

        if self.cuda_cost_compute:
            corr = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=(int(1 + 2 * 3), int(1 + 2 * 3)),
                stride=1,
                padding=0,
                dilation_patch=1,
            )
            cost = corr(x, y)

        else:

            for i in range(2 * self.max_u + 1):

                ind = i - self.max_u
                for j in range(2 * self.max_v + 1):

                    indd = j - self.max_v
                    cost[
                        :,
                        :c,
                        i,
                        j,
                        max(0, -indd) : height - indd,
                        max(0, -ind) : width - ind,
                    ] = x[
                        :, :, max(0, -indd) : height - indd, max(0, -ind) : width - ind
                    ]
                    cost[
                        :,
                        c:,
                        i,
                        j,
                        max(0, -indd) : height - indd,
                        max(0, -ind) : width - ind,
                    ] = y[
                        :, :, max(0, +indd) : height + indd, max(0, ind) : width + ind
                    ]

        if self.remove_warp_hole:

            valid_mask = cost[:, c:, ...].sum(dim=1) != 0
            valid_mask = valid_mask.detach()
            cost = cost * valid_mask.unsqueeze(1).float()

        cost = cost.permute([0, 2, 3, 1, 4, 5]).contiguous()
        cost = cost.view(x.size()[0] * size_u * size_v, c * 2, x.size()[2], x.size()[3])

        cost = self.matching_net(cost)

        cost = cost.view(x.size()[0], size_u, size_v, 1, x.size()[2], x.size()[3])
        cost = cost.permute([0, 3, 1, 2, 4, 5]).contiguous()

        return cost


@SIMILARITY_REGISTRY.register()
class MatryoshkaDilatedCostVolume(nn.Module):
    @configurable
    def __init__(
        self,
        num_groups=1,
        max_displacement=4,
        stride=1,
        dilations=[1, 2, 3, 5, 9, 16],
        use_relu=False,
    ):
        super(MatryoshkaDilatedCostVolume, self).__init__()
        self.num_groups = num_groups
        self.use_relu = use_relu
        self._set_concentric_offsets(dilations=dilations, radius=max_displacement)

        self.corr_layers = nn.ModuleList()

        search_range = 2 * max_displacement + 1
        for i in range(len(dilations)):
            self.corr_layers.append(
                SpatialCorrelationSampler(
                    patch_size=search_range,
                    stride=stride,
                    padding=0,
                    dilation_patch=dilations[i],
                )
            )

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_groups": cfg.NUM_GROUPS,
            "max_displacement": cfg.MAX_DISPLACEMENT,
            "stride": cfg.STRIDE,
            "dilations": cfg.DILATIONS,
            "use_relu": cfg.USE_RELU,
        }

    def _set_concentric_offsets(self, dilations, radius):
        offsets_list = []
        for dilation_i in dilations:
            offsets_i = np.arange(-radius, radius + 1) * dilation_i
            offsets_list.append(offsets_i.tolist())

        offsets = np.array(offsets_list)
        self.register_buffer("offsets", torch.Tensor(offsets).float())

    def get_relative_offsets(self):
        return self.offsets

    def get_search_range(self):
        return self.offsets.shape[1]

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        assert c % self.num_groups == 0
        channels_per_group = c // self.num_groups

        x1 = x1.view(b * self.num_groups, channels_per_group, h, w)
        x2 = x2.view(b * self.num_groups, channels_per_group, h, w)
        cost_list = []

        for corr_fn in self.corr_layers:
            cost = corr_fn(x1, x2)
            _, u, v, h, w = cost.shape
            cost_list.append(cost.view(b, self.num_groups, u, v, h, w))

        cost = torch.cat(cost_list, dim=1)

        if self.use_relu:
            cost = F.leaky_relu(cost, negative_slope=0.1)

        return cost


@SIMILARITY_REGISTRY.register()
class MatryoshkaDilatedCostVolumeList(nn.Module):
    @configurable
    def __init__(
        self,
        num_groups=1,
        max_displacement=4,
        encoder_output_strides=[2, 8],
        dilations=[[1], [1, 2, 3, 5, 9, 16]],
        normalize_feat_l2=False,
        use_relu=False,
    ):
        super(MatryoshkaDilatedCostVolumeList, self).__init__()

        self.normalize_feat_l2 = normalize_feat_l2
        self.cost_volume_list = nn.ModuleList()
        offsets = None

        for idx, (dilations_i, feat_stride_i) in enumerate(
            zip(dilations, encoder_output_strides)
        ):
            assert feat_stride_i <= 8
            cost_volume_i = MatryoshkaDilatedCostVolume(
                num_groups=num_groups,
                max_displacement=max_displacement,
                dilations=dilations_i,
                stride=8 // feat_stride_i,
                use_relu=use_relu,
            )

            self.cost_volume_list.append(cost_volume_i)
            if offsets is None:
                offsets = cost_volume_i.get_relative_offsets() * feat_stride_i
            else:
                offsets = torch.cat(
                    (offsets, cost_volume_i.get_relative_offsets() * feat_stride_i),
                    dim=0,
                )

        self.offsets = offsets
        self._set_global_flow_offsets()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_groups": cfg.NUM_GROUPS,
            "max_displacement": cfg.MAX_DISPLACEMENT,
            "encoder_output_strides": cfg.ENCODER_OUTPUT_STRIDES,
            "dilations": cfg.DILATIONS,
            "normalize_feat_l2": cfg.NORMALIZE_FEAT_L2,
            "use_relu": cfg.USE_RELU,
        }

    def _set_global_flow_offsets(self):
        # process offsets
        num_dilations, search_range = self.offsets.shape
        offsets_2d = torch.zeros((num_dilations, search_range, search_range, 2))
        for idx in range(num_dilations):
            offsets_i, offsets_j = torch.meshgrid(
                self.offsets[idx], self.offsets[idx], indexing="ij"
            )
            offsets_2d[idx, :, :, 0] = offsets_i  # y
            offsets_2d[idx, :, :, 1] = offsets_j  # x

        self.register_buffer("offsets_2d", torch.Tensor(offsets_2d).float())

    def get_global_flow_offsets(self):
        return self.offsets_2d

    def get_search_range(self):
        return self.cost_volume_list[0].get_search_range()

    def forward(self, x1, x2):
        # B, C, U, V, H, W
        cost_list = []
        for idx in range(len(x1)):
            x1_i = x1[idx]
            x2_i = x2[idx]
            if self.normalize_feat_l2:
                x1_i = x1_i / (x1_i.norm(dim=1, keepdim=True) + 1e-9)
                x2_i = x2_i / (x2_i.norm(dim=1, keepdim=True) + 1e-9)
            cost_i = self.cost_volume_list[idx](x1_i, x2_i)
            cost_list.append(cost_i)

        cost = torch.cat(cost_list, dim=1)

        return cost
