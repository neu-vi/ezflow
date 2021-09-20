import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except:
    pass

from ..modules import ConvNormRelu


class Conv2DMatching(nn.Module):
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

    def forward(self, x):

        x = self.matching_net(x)

        return x


class Custom2DConvMatching(nn.Module):
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

    def forward(self, x):

        x = self.matching_net(x)

        return x


class LearnableMatchingCost(nn.Module):
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
