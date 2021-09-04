import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    from spatial_correlation_sampler import SpatialCorrelationSampler

from ..common import ConvNormRelu


class Conv2DMatching(nn.Module):
    def __init__(self, config=(64, 96, 128, 128, 64, 32, 1)):
        super(Conv2DMatching, self).__init__()

        self.match = nn.ModuleList()
        for i in range(len(config) - 1):
            self.match.append(ConvNormRelu(config[i], config[i + 1]))

        self.match = nn.Sequential(*self.match)

    def forward(self, x):

        x = self.match(x)

        return x


class LearnableMatchingCost(nn.Module):
    def __init__(
        self,
        max_u,
        max_v,
        remove_warp_hole=True,
        cuda_cost_compute=False,
        matching_net=None,
    ):
        super(LearnableMatchingCost, self).__init__()

        if matching_net is not None:
            self.matching_net = matching_net
        else:
            self.matching_net = Conv2DMatching()

        self.max_u = max_u
        self.max_v = max_v
        self.remove_warp_hole = remove_warp_hole
        self.cuda_cost_compute = cuda_cost_compute

    def forward(self, x, y):

        size_u = 2 * self.max_u + 1
        size_v = 2 * self.max_v + 1
        b, c, height, width = x.shape

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
