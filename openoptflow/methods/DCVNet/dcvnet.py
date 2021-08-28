import torch
import torch.nn as nn
import torch.nn.functional as F
from openopticalflow.decoder import Convolution3DJointFusion
from openopticalflow.encoder import BasicEncoder

from .utils import replace_bn, replace_relu


class DeepLabV3OpticalFlowEstimator(nn.Module):
    def __init__(self, cfg, criterion=None):
        super(DeepLabV3OpticalFlowEstimator, self).__init__()

        self.criterion = criterion
        self.cfg = cfg

        # elif cfg.MODEL.ENCODER == 'raft_stride28_256d':
        batch_norm = cfg.MODEL.FEATURE_ENCODER_NORM
        assert batch_norm in ["instance", "batch"]

        base = RAFT_Stride28_256d(norm_fn=batch_norm)
        self.base = base

        self.flow_decoder = Convolution3DJointFusion(
            cfg, feat_stride=base.output_strides, app_feat_dim=base.output_dims
        )

        if cfg.MODEL.INIT_WEIGHTS:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(
                    m,
                    (
                        nn.BatchNorm2d,
                        nn.InstanceNorm2d,
                        nn.GroupNorm,
                        nn.BatchNorm3d,
                        nn.SyncBatchNorm,
                    ),
                ):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if cfg.MODEL.USE_INST_NORM:
            self.flow_decoder = replace_bn(self.flow_decoder, "instance_norm")

        self.use_leaky_relu = cfg.MODEL.USE_LEAKY_RELU
        if cfg.MODEL.USE_LEAKY_RELU:
            self = replace_relu(self)

    def freeze_encoder_bn(self):
        for m in self.base.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def freeze_decoder_bn(self):
        for m in self.flow_decoder.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                m.eval()

    def reset_bn_running_stats(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                m.reset_running_stats()

    def reset_bn_parameters(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                m.reset_parameters()

    def get_image_encoding(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_before_aspp = self.layer4(x_tmp)
        x = x_before_aspp

        return x, x_tmp, x_before_aspp

    def forward(
        self,
        x,
        nb_x,
        y=None,
        valid=None,
        output_entropy=False,
        output_intermediate=False,
    ):
        x_size = x.size()
        h, w = x.shape[2:]

        x = self.base(x)
        nb_x = self.base(nb_x)

        if isinstance(x, tuple):
            assert isinstance(nb_x, tuple)
            nb_x = nb_x[0]
            cxt_x = x[1]
            x = x[0]
        else:
            cxt_x = x

        try:
            need_relu_for_context = self.base.need_relu_for_context()
        except:
            need_relu_for_context = True

        if need_relu_for_context:
            if self.use_leaky_relu:
                cxt_x = [F.leaky_relu(cxt_x_i, negative_slope=0.1) for cxt_x_i in cxt_x]
            else:
                cxt_x = [F.relu(cxt_x_i) for cxt_x_i in cxt_x]

        flow = self.flow_decoder(
            x,
            nb_x,
            cxt_x,
            output_entropy=output_entropy,
            output_intermediate=output_intermediate,
        )

        if self.training:

            if isinstance(flow, tuple) or isinstance(flow, list):

                if self.cfg.MODEL.DECODER == "dcv3d_dilation_joint_fusion_aux_loss":
                    loss_list = []

                    for flow_i in flow:
                        loss_i = self.criterion(flow_i, y, valid)
                        loss_list.append(loss_i)

                    if len(loss_list) < 3:
                        loss_list = [None] + loss_list

                    assert len(loss_list) == 3, [len(loss_list), len(flow)]

                    loss_list = loss_list[::-1]

                    return flow[-1], loss_list[0], loss_list[1], loss_list[2]

                else:

                    if self.cfg.MODEL.CONVEX_AS_RESIDUAL:
                        flow, coarse_flow = flow
                        main_loss = self.criterion(flow, y, valid)
                        aux_loss = self.criterion(coarse_flow, y, valid)
                    else:
                        flow, flow_entropy, fusion_entropy = flow
                        assert self.criterion is not None

                        main_loss = self.criterion(flow, y, valid)
                        aux_loss1 = torch.mean(flow_entropy)

                        if fusion_entropy is not None:
                            aux_loss2 = torch.mean(fusion_entropy)
                        else:
                            aux_loss2 = None

                    return flow, main_loss, aux_loss1, aux_loss2
            else:

                if self.criterion is not None:
                    main_loss = self.criterion(flow, y, valid)
                else:
                    main_loss = None

                return flow, main_loss

        else:
            return flow
