import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....similarity import GroupWiseCostVolume3D
from .....utils import convex_upsample_flow
from .flow_fusion import ConvexUpMaskConv2D
from .unet_filter import UNetGroupsASPP


class Convolution3DJointFusion:
    def __init__(
        self,
        feat_stride=8,
        app_feat_dim=128,
        search_radius=4,
        cost_volume_num_groups=4,
        cost_volume_proj_dim=128,
        dilations=[[1], [1, 3, 5, 9, 13, 21]],
        p_dropout=0.1,
        unique_flow_grids=True,
        detach_flow_entropy=False,
        cv_spatial_pooling=-1,
        num_hypotheses=1,  # check this, what's going on with -1
        truncated=False,
        feature_l2_normalize=True,
        aux_loss_weight=[],
        relu_after_corr=False,
        use_tail_op=False,
        nln_pos=[],
        unet_inter_dim=48,
        use_cbam=False,
        use_dec_app_feat_pool=False,
        use_heavy_stem=False,
        use_se=False,
        cbam_spatial=True,
        cbam_reduction=16,
        use_stride2_feat=False,
        pool_stride2_feat=False,
        mask_kernel_size=3,
    ):
        super(Convolution3DJointFusion, self).__init__()

        if cv_spatial_pooling < 0:
            cv_spatial_pooling = None

        # if num_hypotheses <= 0:
        #     num_hypotheses = 1

        self.num_hypotheses = num_hypotheses
        self.wsize = 3

        self.cost_volume_num_groups = cost_volume_num_groups

        self.detach_flow_entropy = detach_flow_entropy

        if isinstance(dilations[0], int):
            dilations = [dilations]
        self.dilations = dilations

        self.feat_stride = feat_stride

        self.output_aux = len(aux_loss_weight) > 0

        self.feature_l2_normalize = feature_l2_normalize

        search_range = 2 * search_radius + 1
        self.search_range = search_range
        self.cost_volume_list = []

        for dilations_i, feat_stride_i in zip(dilations, feat_stride):

            cost_volume_i = GroupWiseCostVolume3D(
                num_groups=cost_volume_num_groups,
                search_range=search_range,
                dilations=dilations_i,
                pool_scales=cv_spatial_pooling,
                stride=8 // feat_stride_i,
                use_bn=False,
                relu_after_corr=relu_after_corr,
                use_tail_op=use_tail_op,
            )
            self.cost_volume_list.append(cost_volume_i)
        self.cost_volume_list = nn.ModuleList(self.cost_volume_list)

        num_dilations = 0
        for dilations_i in dilations:
            num_dilations += len(dilations_i)

        self.num_dilations = num_dilations
        flow_grids_list = []

        for dilations_i, feat_stride_i in zip(dilations, feat_stride):
            for dl in dilations_i:
                flow_grids = self.get_flow_grids(search_radius, feat_stride_i, dl)
                flow_grids_list.append(torch.Tensor(flow_grids))

        self.register_buffer("flow_grids_list", torch.stack(flow_grids_list, 0))

        cost_volume_dim = cost_volume_num_groups * num_dilations
        if cv_spatial_pooling is not None:
            if isinstance(cv_spatial_pooling, int):
                cost_volume_dim = cost_volume_dim * (1 + cv_spatial_pooling)
            elif isinstance(cv_spatial_pooling, tuple):
                cost_volume_dim = cost_volume_dim * (1 + len(cv_spatial_pooling))
            else:
                raise ValueError(
                    "Not supported cost volume spatial pooling: {}".format(
                        cv_spatial_pooling
                    )
                )

        spatial_dim = (search_range ** 2 + 1) // 2
        mid_dim = cost_volume_proj_dim
        out_dim = num_dilations * self.num_hypotheses

        self.cv_filter = UNetGroupsASPP(
            cost_volume_dim,
            spatial_dim,
            mid_dim,
            out_dim,
            unet_inter_dim,
            nln_pos,
            use_cbam,
            cbam_reduction,
            cbam_spatial,
            use_se,
            self.num_dilations,
            use_heavy_stem,
        )

        not_process_app_feat = True

        self.app_feat_processors = []
        for feat_stride_i, app_feat_dim_i in zip(feat_stride, app_feat_dim):

            downsample_factor = 8 // feat_stride_i

            if downsample_factor > 1 and (not not_process_app_feat):

                if use_dec_app_feat_pool:
                    processor = nn.AvgPool2d(
                        kernel_size=downsample_factor, stride=downsample_factor
                    )
                else:
                    processor = nn.Sequential(
                        nn.Conv2d(
                            app_feat_dim_i,
                            app_feat_dim_i,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            app_feat_dim_i,
                            app_feat_dim_i,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                        ),
                        nn.ReLU(),
                    )

            else:
                processor = None
            self.app_feat_processors.append(processor)
        self.app_feat_processors = nn.ModuleList(self.app_feat_processors)

        self.fusion_decoder_type = "plain2d_nobnv2_dilation_noapp"

        in_planes = (2 + 1) * num_dilations * self.num_hypotheses
        out_planes = num_dilations * self.num_hypotheses * 2

        use_stride2_feat = use_stride2_feat and len(feat_stride) > 1

        try:
            cv_filter_feat_dim = self.cv_filter.feat_dim()
        except:
            cv_filter_feat_dim = 0

        self.flow_fusion = ConvexUpMaskConv2D(
            in_planes,
            out_planes,
            feat_stride,
            app_feat_dim,
            use_stride2_feat=use_stride2_feat,
            pool_stride2_feat=pool_stride2_feat,
            mask_kernel_size=mask_kernel_size,
            cv_filter_feat_dim=cv_filter_feat_dim,
        )

        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                raise Exception("No BNs are expected in the decoder.")

    def get_flow_grids(self, search_radius, feat_stride, dilation):
        x_grids = np.arange(-search_radius, search_radius + 1) * dilation * feat_stride
        grids = np.meshgrid(x_grids, x_grids)
        grids = np.stack(grids, 2)
        return grids.reshape(1, (2 * search_radius + 1) ** 2 * 2, 1, 1)

    def logits_to_flow_entropy(self, flow_logits, grids):

        b, uv, h, w = flow_logits.shape

        if self.truncated:

            u = v = self.search_range
            old_flow_logtis = flow_logits.view(b, u, v, h, w)
            idx = flow_logits.argmax(1).unsqueeze(1)
            mask = torch.zeros_like(flow_logits)
            mask.scatter_(1, idx, 1)
            mask = mask.view(b, 1, u, v, -1)
            mask = F.max_pool3d(
                mask,
                kernel_size=(self.wsize * 2 + 1, self.wsize * 2 + 1, 1),
                stride=1,
                padding=(self.wsize, self.wsize, 0),
            )
            mask = mask[:, 0].view(b, u, v, h, w)

            ninf = flow_logits.clone().fill_(-np.inf).view(b, u, v, h, w)
            flow_logits = torch.where(mask.byte(), old_flow_logtis, ninf)
            flow_logits = flow_logits.view(b, uv, h, w)

        flow_probs = F.softmax(flow_logits, dim=1)

        num_grids = self.search_range ** 2
        grids = grids.view(1, self.search_range * self.search_range, 2, 1, 1)
        flowx = grids[:, :, 0]
        flowy = grids[:, :, 1]
        flowx = flowx.repeat(b, 1, h, w)
        flowy = flowy.repeat(b, 1, h, w)
        u = torch.sum(flow_probs * flowx, dim=1, keepdim=True)
        v = torch.sum(flow_probs * flowy, dim=1, keepdim=True)
        flow = torch.cat((u, v), dim=1)

        flow_entropy = -flow_probs * torch.clamp(flow_probs, 1e-9, 1 - 1e-9).log()
        flow_entropy = flow_entropy.sum(1, keepdim=True)

        return flow, flow_entropy

    def get_flow_estimation(self, flow_logits):

        b, c, uv, h, w = flow_logits.shape
        flow_logits = flow_logits.view(
            b, self.num_dilations * self.num_hypotheses, -1, h, w
        )

        flow_list = []
        flow_entropy_list = []

        for i in range(self.num_dilations * self.num_hypotheses):

            flow_grids_idx = i // self.num_hypotheses
            flow, flow_entropy = self.logits_to_flow_entropy(
                flow_logits[:, i], self.flow_grids_list[flow_grids_idx]
            )
            flow_list.append(flow)
            flow_entropy_list.append(flow_entropy)

        flow = torch.cat(flow_list, dim=1)
        flow_entropy = torch.cat(flow_entropy_list, dim=1)

        return flow, flow_entropy

    def forward(self, x1, x2, app_x, output_entropy=False, output_intermediate=False):

        cost_list = []
        out_h, out_w = x1[-1].shape[2:]

        for idx in range(len(x1)):

            x1_i = x1[idx]
            x2_i = x2[idx]
            if self.feature_l2_normalize:
                x1_i = x1_i / (x1_i.norm(dim=1, keepdim=True) + 1e-9)
                x2_i = x2_i / (x2_i.norm(dim=1, keepdim=True) + 1e-9)
            cost_i = self.cost_volume_list[idx](x1_i, x2_i)
            cost_list.append(cost_i)

        cost = torch.cat(cost_list, dim=1)

        flow_logits = self.cv_filter(cost)
        if isinstance(flow_logits, tuple) or isinstance(flow_logits, list):
            flow_logits, cv_filter_feat = flow_logits
        else:
            cv_filter_feat = None
        flow, flow_entropy = self.get_flow_estimation(flow_logits)

        processed_app_x = []

        for app_x_i, processor_i in zip(app_x, self.app_feat_processors):
            if processor_i is not None:
                app_x_i_p = processor_i(app_x_i)
            else:
                app_x_i_p = app_x_i
            processed_app_x.append(app_x_i_p)

        final_flow, mask_logits, fusion_probs = self.flow_fusion(
            flow, flow_entropy, cv_filter_feat
        )

        if mask_logits is not None:
            try:
                scale_factor = self.flow_fusion.output_scale_factor()
            except:
                scale_factor = self.feat_stride[-1]

            final_flow = convex_upsample_flow(final_flow, mask_logits, scale_factor)
        else:
            try:
                scale_factor = self.flow_fusion.output_scale_factor()
            except:
                scale_factor = self.feat_stride[-1]
            final_flow = F.interpolate(
                final_flow,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )

        if output_intermediate:
            return (
                final_flow,
                flow_logits,
                self.flow_grids,
            )
        else:
            if self.training and self.output_aux:
                if fusion_probs is not None:
                    fusion_entropy = (
                        -fusion_probs * torch.clamp(fusion_probs, 1e-9, 1 - 1e-9).log()
                    )
                    fusion_entropy = fusion_entropy.sum(1, keepdim=True)
                else:
                    fusion_entropy = None
                return final_flow, flow_entropy, fusion_entropy

            return final_flow
