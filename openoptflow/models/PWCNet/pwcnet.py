import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import build_decoder
from ...encoder import build_encoder
from ...functional import CorrelationLayer
from ...utils import warp
from ..build import MODEL_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )


@MODEL_REGISTRY.register()
class PWCNet(nn.Module):
    def __init__(self, cfg):
        super(PWCNet, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)
        self.correlation_layer = CorrelationLayer(
            pad_size=cfg.SIMILARITY.PAD_SIZE,
            max_displacement=cfg.SIMILARITY.MAX_DISPLACEMENT,
        )

        search_range = (2 * cfg.SIMILARITY.MAX_DISPLACEMENT + 1) ** 2

        self.decoder_layers = nn.ModuleList()
        decoder_cfg = cfg.DECODER.CONFIG

        self.up_feature_layers = nn.ModuleList()

        for i in range(len(decoder_cfg)):

            if i == 0:
                concat_channels = search_range
            else:
                concat_channels = (
                    search_range + decoder_cfg[i] + cfg.SIMILARITY.MAX_DISPLACEMENT,
                )

            to_flow = False if i == len(decoder_cfg) - 1 else True

            self.decoder_layers.append(
                build_decoder(
                    config=decoder_cfg,
                    to_flow=to_flow,
                    concat_channels=concat_channels,
                )
            )

            self.up_feature_layers.append(
                deconv(
                    concat_channels + sum(decoder_cfg),
                    2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

        self.deconv_layers = nn.ModuleList()
        for i in range(len(decoder_cfg)):
            self.deconv_layers.append(deconv(2, 2, kernel_size=4, stride=2, padding=1))

        self.dc_conv = nn.ModuleList(
            [
                conv(
                    search_range
                    + cfg.SIMILARITY.MAX_DISPLACEMENT
                    + decoder_cfg[-1]
                    + sum(decoder_cfg),
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                ),
            ]
        )
        self.dc_conv.append(
            conv(
                decoder_cfg[0],
                decoder_cfg[0],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            )
        )

        padding = 4
        dilation = 4
        for i in range(len(decoder_cfg) - 2):
            self.dc_conv.append(
                conv(
                    decoder_cfg[i],
                    decoder_cfg[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                )
            )
            padding *= 2
            dilation *= 2

        self.dc_conv.append(
            conv(
                decoder_cfg[3],
                decoder_cfg[4],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            )
        )
        self.dc_conv.append(
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.dc_conv = nn.Sequential(*self.dc_conv)

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

    def _corr_relu(self, features1, features2):

        corr = self.correlation_layer(features1, features2)

        return F.leaky_relu(corr, negative_slope=0.1)

    def forward(self, img1, img2):

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        up_flow, up_features = None, None
        up_flow_scale = 0.625

        flow_preds = []

        for i in range(len(feature_pyramid1)):

            if i == 0:
                corr = self._corr_relu(feature_pyramid1[i], feature_pyramid2[i])
                concatenated_features = corr

            else:

                warped_features = warp(feature_pyramid2[i], up_flow * up_flow_scale)
                up_flow_scale *= 2

                corr = self._corr_relu(feature_pyramid1[i], warped_features)

                concatenated_features = torch.cat(
                    [corr, feature_pyramid1[i], up_flow, up_features], dim=1
                )

            flow = self.decoder_layers[i](concatenated_features)
            flow_preds.append(flow)

            up_flow = self.deconv_layers[i](flow)
            up_features = self.up_feature_layers[i](concatenated_features)

        flow_preds.reverse()

        features = flow_preds[0]
        flow = nn.Conv2d(
            features.shape[1], 2, kernel_size=3, stride=1, padding=1, bias=True
        )(features)
        flow += self.dc_conv(features)
        flow_preds[0] = flow

        if self.training:
            return flow_preds

        else:
            return flow
