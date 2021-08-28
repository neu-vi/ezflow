import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupWiseCorrelation(nn.Module):
    def __init__(self, patch_size, dilation, num_groups, stride, relu_after_corr=True):
        super(GroupWiseCorrelation, self).__init__()

        self.corr = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=patch_size,
            stride=stride,
            padding=0,
            dilation_patch=dilation,
        )
        self.num_groups = num_groups
        self.relu_after_corr = relu_after_corr

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        assert c % self.num_groups == 0
        channels_per_group = c // self.num_groups
        x1 = x1.view(b, channels_per_group, self.num_groups, h, w)
        x2 = x2.view(b, channels_per_group, self.num_groups, h, w)
        cost = []
        for i in range(self.num_groups):
            x1_i = x1[:, :, i].contiguous()
            x2_i = x2[:, :, i].contiguous()
            cost_i = self.corr(x1_i, x2_i)
            if self.relu_after_corr:
                cost_i = F.leaky_relu(cost_i, negative_slope=0.1)
            cost.append(cost_i)
        return torch.stack(cost, 1)


class GroupWiseCostVolume3D(nn.Module):
    def __init__(
        self,
        num_groups=16,
        search_range=9,
        dilations=[1, 2, 4, 8],
        pool_scales=None,
        stride=1,
        use_bn=True,
        relu_after_corr=True,
        use_tail_op=True,
    ):
        super(GroupWiseCostVolume3D, self).__init__()

        self.corr_ops = []
        self.corr_tail_ops = []
        for dilation_factor in dilations:
            corr_op = GroupWiseCorrelation(
                (search_range, search_range),
                dilation_factor,
                num_groups,
                stride,
                relu_after_corr,
            )
            if use_tail_op:
                if use_bn:
                    corr_tail_op = nn.Sequential(
                        nn.Conv3d(num_groups, num_groups, kernel_size=3, padding=1),
                        nn.BatchNorm3d(num_groups),
                        nn.ReLU(),
                    )
                else:
                    corr_tail_op = nn.Sequential(
                        nn.Conv3d(num_groups, num_groups, kernel_size=3, padding=1),
                        nn.ReLU(),
                    )
            else:
                corr_tail_op = None
            self.corr_ops.append(corr_op)
            self.corr_tail_ops.append(corr_tail_op)
        self.corr_ops = nn.ModuleList(self.corr_ops)
        self.corr_tail_ops = nn.ModuleList(self.corr_tail_ops)
        # spatial pooling
        self.pool_scales = pool_scales
        self.search_range = search_range

    def get_cost_volume_single_scale(self, corr_op, corr_tail_op, x, nb_x):
        cost = corr_op(x, nb_x)
        b, c, u, v, h, w = cost.shape
        cost = cost.view(b, c, (u * v), h, w)
        if corr_tail_op is not None:
            cost = corr_tail_op(cost)
        return cost

    def spatial_pooling_power_of_2(self, cost):
        b, c, u, v = cost.shape
        cost_pooled = [cost]
        num_pooling = self.pool_scales
        for ps in range(num_pooling):
            cost = F.avg_pool2d(cost, 2, stride=2)
            cost_p = F.interpolate(
                cost, size=(u, v), mode="bilinear", align_corners=False
            )
            cost_pooled.append(cost_p)
        cost = torch.cat(cost_pooled, 1)
        return cost

    def spatial_pooling_pool_scales(self, cost):
        b, c, u, v = cost.shape
        cost_pooled = [cost]
        for ps in self.pool_scales:
            cost = F.adaptive_avg_pool2d(cost, ps)
            cost_p = F.interpolate(
                cost, size=(u, v), mode="bilinear", align_corners=False
            )
            cost_pooled.append(cost_p)
        cost = torch.cat(cost_pooled, 1)
        return cost

    def get_cost_volume(self, x, nb_x):
        cost_list = []
        for corr_op, corr_tail_op in zip(self.corr_ops, self.corr_tail_ops):
            cost = self.get_cost_volume_single_scale(corr_op, corr_tail_op, x, nb_x)
            cost_list.append(cost)
        cost = torch.cat(cost_list, 1)

        if self.pool_scales is not None:
            b, c, uv, h, w = cost.shape
            cost = cost.view(b, c, self.search_range, self.search_range, h, w)
            # b, h, w, c, u, v
            cost = cost.permute(0, 4, 5, 1, 2, 3).contiguous()
            cost = cost.view((b * h * w), c, self.search_range, self.search_range)
            if isinstance(self.pool_scales, tuple):
                cost = self.spatial_pooling_pool_scales(cost)
            elif isinstance(self.pool_scales, int):
                cost = self.spatial_pooling_power_of_2(cost)
            else:
                raise RuntimeError(
                    "Not supported pool_scales: {}".format(type(self.pool_scales))
                )
            c = cost.shape[1]
            cost = cost.view(b, h, w, c, self.search_range, self.search_range)
            cost = cost.permute(0, 3, 4, 5, 1, 2).contiguous()
            cost = cost.view(b, c, (self.search_range ** 2), h, w)
        return cost

    def forward(self, x, nb_x):
        cost = self.get_cost_volume(x, nb_x)

        return cost
