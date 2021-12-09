import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import ConvDecoder
from ..encoder import build_encoder
from ..modules import conv, deconv
from ..similarity import CorrelationLayer
from ..utils import warp
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PWCNet(nn.Module):
    """
    Implementation of the paper
    `PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume <https://arxiv.org/abs/1709.02371>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

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
                    search_range + decoder_cfg[i] + cfg.SIMILARITY.MAX_DISPLACEMENT
                )

            self.decoder_layers.append(
                ConvDecoder(
                    config=decoder_cfg,
                    to_flow=True,
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
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

    def _corr_relu(self, features1, features2):

        corr = self.correlation_layer(features1, features2)

        return F.leaky_relu(corr, negative_slope=0.1)

    def forward(self, img1, img2):
        """
        Performs forward pass of the network

        Parameters
        ----------
        img1 : torch.Tensor
            Image to predict flow from
        img2 : torch.Tensor
            Image to predict flow to

        Returns
        -------
        torch.Tensor
            Flow from img1 to img2
        """

        H, W = img1.shape[-2:]

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        up_flow, up_features = None, None
        up_flow_scale = 0.625

        flow_preds = []

        for i in range(len(self.decoder_layers)):

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

            flow, features = self.decoder_layers[i](concatenated_features)
            flow_preds.append(flow)

            up_flow = self.deconv_layers[i](flow)
            up_features = self.up_feature_layers[i](features)

        flow_preds.reverse()
        flow_preds[0] += self.dc_conv(features)

        if self.training:
            return flow_preds

        else:

            flow = flow_preds[0]

            if self.cfg.INTERPOLATE_FLOW:

                H_, W_ = flow.shape[-2:]
                flow = F.interpolate(
                    flow, img1.shape[-2:], mode="bilinear", align_corners=True
                )
                flow_u = flow[:, 0, :, :] * (W / W_)
                flow_v = flow[:, 1, :, :] * (H / H_)
                flow = torch.stack([flow_u, flow_v], dim=1)

            if self.cfg.FLOW_SCALE_FACTOR is not None:
                flow *= self.cfg.FLOW_SCALE_FACTOR

            return flow
