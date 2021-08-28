import torch
import torch.nn as nn
import torch.nn.functional as F

from .....utils import convex_upsample_flow


class ConvexUpMaskConv2D(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        feat_stride,
        app_feat_dim,
        use_stride2_feat=True,
        pool_stride2_feat=False,
        mask_kernel_size=1,
        cv_filter_feat_dim=0,
    ):
        super(ConvexUpMaskConv2D, self).__init__()

        fusion_in_planes = in_planes + cv_filter_feat_dim
        self.fusion_body = nn.Sequential(
            self._conv_relu(fusion_in_planes, 128),
            self._conv_relu(128, 128, padding=2, dilation=2),
            self._conv_relu(128, 128, padding=4, dilation=4),
            self._conv_relu(128, 96, padding=8, dilation=8),
        )
        self.fusion_head = nn.Sequential(
            self._conv_relu(96, 64, padding=16, dilation=16),
            self._conv_relu(64, 32),
            nn.Conv2d(32, out_planes, kernel_size=3, padding=1),
        )

        out_stride = 4
        self.mask_head = nn.Sequential(
            self._conv_relu(96, 64, padding=16, dilation=16),
            self._conv_relu(64, 64),
        )
        self.mask_tail = nn.Conv2d(
            64,
            out_stride ** 2 * 9,
            kernel_size=mask_kernel_size,
            padding=(mask_kernel_size - 1) // 2,
        )

        mask2_in_planes = 2 + 64 + cv_filter_feat_dim
        out_stride = 2
        self.mask_head2 = nn.Sequential(
            self._conv_relu(mask2_in_planes, 128),
            self._conv_relu(128, 64),
            nn.Conv2d(
                64,
                out_stride ** 2 * 9,
                kernel_size=mask_kernel_size,
                padding=(mask_kernel_size - 1) // 2,
            ),
        )

    def output_scale_factor(self):
        return 2

    def _conv_relu(
        in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1
    ):

        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, flow, flow_entropy, cv_filter_feat=None):

        if cv_filter_feat is not None:
            fusion_x = torch.cat((flow, flow_entropy, cv_filter_feat), dim=1)
        else:
            fusion_x = torch.cat((flow, flow_entropy), dim=1)
        fusion_x = self.fusion_body(fusion_x)
        fusion_logits = self.fusion_head(fusion_x)
        mask_x = self.mask_head(fusion_x)
        mask_logits = self.mask_tail(mask_x)

        b, c, h, w = fusion_logits.shape
        fusion_logits = fusion_logits.view(b, -1, 2, h, w)
        fusion_probs = F.softmax(fusion_logits, dim=1)
        final_flow = torch.sum(flow.view(b, -1, 2, h, w) * fusion_probs, dim=1)
        final_flow = convex_upsample_flow(final_flow, mask_logits, 4)

        mask_x = convex_upsample_flow(mask_x, mask_logits, 4)

        if cv_filter_feat is not None:
            cv_filter_feat = convex_upsample_flow(cv_filter_feat, mask_logits, 4)
            mask_x = torch.cat((final_flow, mask_x, cv_filter_feat), dim=1)
        else:
            mask_x = torch.cat((final_flow, mask_x), dim=1)
        mask_logits2 = self.mask_head2(mask_x)

        # final_flow = F.interpolate(final_flow, mode='bilinear', scale_factor=4, align_corners=False)

        return final_flow, mask_logits2, fusion_probs
