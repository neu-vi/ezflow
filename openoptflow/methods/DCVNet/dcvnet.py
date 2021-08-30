import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import Convolution3DJointFusion
from ...encoder import BasicEncoder
from .utils import replace_bn, replace_relu


class DCVNet(nn.Module):  # DeepLabV3OpticalFlowEstimator
    def __init__(
        self,
        norm="batch",
        feature_strides=8,
        features_dim=128,
        leaky_relu=True,
        **kwargs
    ):
        super(DCVNet, self).__init__()

        base = BasicEncoder(norm=norm)
        self.base = base

        self.flow_decoder = Convolution3DJointFusion(
            feat_stride=feature_strides, app_feat_dim=features_dim, **kwargs
        )

        self._init_weights()

        if norm == "instance":
            self.flow_decoder = replace_bn(self.flow_decoder, "instance_norm")

        self.leaky_relu = leaky_relu
        if self.leaky_relu:
            self = replace_relu(self)

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

        return flow
