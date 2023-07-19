import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import configurable
from .build import MODULE_REGISTRY


def _get_norm_fn(in_dim, norm_fn="instance"):
    assert norm_fn in ["instance", "batch", "none"]
    if norm_fn == "instance":
        return nn.InstanceNorm2d(in_dim)
    elif norm_fn == "batch":
        return nn.BatchNorm2d(in_dim)
    elif norm_fn == "none":
        return nn.Identity()


@MODULE_REGISTRY.register()
class ASPPModule2D(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self,
        features,
        inner_features=512,
        out_features=512,
        dilations=(4, 8, 16),
        groups=1,
        norm_fn="none",
    ):
        super(ASPPModule2D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                inner_features * 5,
                out_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=groups,
            ),
            _get_norm_fn(inner_features, norm_fn),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=False
        )

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


@MODULE_REGISTRY.register()
class UNet2DASPP(nn.Module):
    def __init__(
        self,
        in_planes,
        inter_planes,
        out_planes,
        use_heavy_stem,
        num_groups=1,
        norm_fn="none",
    ):
        super(UNet2DASPP, self).__init__()

        if use_heavy_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    inter_planes,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=num_groups,
                ),
                _get_norm_fn(inter_planes, norm_fn),
                nn.ReLU(),
                nn.Conv2d(
                    inter_planes,
                    inter_planes,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=num_groups,
                ),
                _get_norm_fn(inter_planes, norm_fn),
                nn.ReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    inter_planes,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=num_groups,
                ),
                _get_norm_fn(inter_planes, norm_fn),
                nn.ReLU(),
            )

        # in 1/2, out: 1/4
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                inter_planes,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=2,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
            nn.Conv2d(
                inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
        )

        # in: 1/4, out: 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=2,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
            nn.Conv2d(
                inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
        )

        # in: 1/8, out : 1/8
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
            # nn.Conv2d(inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            ASPPModule2D(
                inter_planes * 2,
                2 * inter_planes,
                inter_planes * 2,
                dilations=(2, 4, 8),
                groups=num_groups,
                norm_fn=norm_fn,
            ),
        )

        # in: 1/8, out: 1/4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                inter_planes * 2 + inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
            nn.Conv2d(
                inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes * 2, norm_fn),
            nn.ReLU(),
        )

        # in: 1/4, out: 1/2
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                inter_planes * 2 + inter_planes,
                inter_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes, norm_fn),
            nn.ReLU(),
            nn.Conv2d(
                inter_planes,
                inter_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups,
            ),
            _get_norm_fn(inter_planes, norm_fn),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.stem(x)

        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x3 = F.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x3 = torch.cat((x3, x1), dim=1)
        x4 = self.conv4(x3)

        x4 = F.interpolate(x4, size=x.shape[2:], mode="bilinear", align_corners=False)
        x4 = torch.cat((x4, x), dim=1)
        x5 = self.conv5(x4)
        return x5


@MODULE_REGISTRY.register()
class DCVFilter_UNet_NoBN_ASPP2D_GroupConvStem_Joint(nn.Module):
    def __init__(
        self,
        cv_num_groups,
        num_dilations,
        cv_search_range,
        feat_in_planes,
        out_planes,
        inter_planes,
        use_heavy_stem,
        stem_use_group_conv,
        norm_fn,
        **kwargs
    ):
        super(DCVFilter_UNet_NoBN_ASPP2D_GroupConvStem_Joint, self).__init__()

        in_planes = cv_num_groups * num_dilations * (cv_search_range**2)

        stem_num_groups = num_dilations
        if not stem_use_group_conv:
            stem_num_groups = 1
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_planes,
                2 * in_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=stem_num_groups,
            ),
            _get_norm_fn(2 * in_planes, norm_fn),
            nn.ReLU(),
            nn.Conv2d(
                2 * in_planes,
                in_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=stem_num_groups,
            ),
            _get_norm_fn(in_planes, norm_fn),
            nn.ReLU(),
        )
        if in_planes != out_planes:
            self.stem_xform = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1), nn.ReLU()
            )
        else:
            self.stem_xform = None

        # stage1
        in_planes_stage1 = out_planes + feat_in_planes
        self.unet_stage1 = UNet2DASPP(
            in_planes_stage1, inter_planes, out_planes, use_heavy_stem, norm_fn=norm_fn
        )
        self.flow_out_stage1 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, padding=1
        )
        self.up_out_stage1 = nn.Conv2d(
            inter_planes, 8**2 * 9, kernel_size=3, padding=1
        )

    def forward(self, cost, x, output_proj=False):
        # we use the feature from the stride of 8 only for now
        x = x[-1]

        b, c, u, v, h, w = cost.shape
        cost = cost.view(b, c * u * v, h, w)

        cost = self.stem(cost)
        if self.stem_xform is not None:
            cost = self.stem_xform(cost)
        flow_logits_stage0 = cost

        # stage 1
        x_stage1 = torch.cat((x, cost), dim=1)
        x_stage1 = self.unet_stage1(x_stage1)
        flow_logits_stage1 = self.flow_out_stage1(x_stage1)
        flow_logits_stage1 = flow_logits_stage1 + flow_logits_stage0
        up_logits_stage1 = self.up_out_stage1(x_stage1)

        # stage 2
        flow_logits_stage2 = None
        up_logits_stage2 = None

        return (flow_logits_stage0, flow_logits_stage1, flow_logits_stage2), (
            None,
            up_logits_stage1,
            up_logits_stage2,
        )
