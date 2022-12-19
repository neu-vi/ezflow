import torch
import torch.nn as nn

from ..config import configurable
from ..modules import deconv
from ..similarity import IterSpatialCorrelationSampler as SpatialCorrelationSampler
from ..utils import warp
from .build import DECODER_REGISTRY
from .conv_decoder import ConvDecoder


@DECODER_REGISTRY.register()
class PyramidDecoder(nn.Module):
    """
    Applies a 2D Convolutional decoder to regress the optical flow
    from the intermediate outputs convolutions of the encoder.
    Used in **PWCNet** (https://arxiv.org/abs/1709.02371)

    Parameters
    ----------
    config : List[int], default : [128, 128, 96, 64, 32]
        List containing all output channels of the decoder.
    to_flow : bool, default : True
        If True, regresses the flow of shape N x 2 x H x W.
    max_displacement: int, default: 4
        Maximum displacement for cost volume computation.
    pad_size: int, default: 0
        Pad size for cost volume computation.
    flow_scale_factor: float, default: 20.0
        Scale factor for upscaling flow predictions.
    """

    @configurable
    def __init__(
        self,
        config=[128, 128, 96, 64, 32],
        to_flow=True,
        max_displacement=4,
        pad_size=0,
        flow_scale_factor=20.0,
    ):
        super(PyramidDecoder, self).__init__()
        self.config = config
        self.flow_scale_factor = flow_scale_factor

        self.correlation_layer = SpatialCorrelationSampler(
            kernel_size=1, patch_size=2 * max_displacement + 1, padding=pad_size
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        search_range = (2 * max_displacement + 1) ** 2

        self.decoder_layers = nn.ModuleList()

        self.up_feature_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        for i in range(len(config)):

            if i == 0:
                concat_channels = search_range
            else:
                concat_channels = search_range + config[i] + max_displacement

            self.decoder_layers.append(
                ConvDecoder(
                    config=config,
                    to_flow=to_flow,
                    concat_channels=concat_channels,
                )
            )

            if i < len(config) - 1:
                self.deconv_layers.append(
                    deconv(2, 2, kernel_size=4, stride=2, padding=1)
                )

                self.up_feature_layers.append(
                    deconv(
                        concat_channels + sum(config),
                        2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

    @classmethod
    def from_config(self, cfg):
        return {
            "config": cfg.CONFIG,
            "to_flow": cfg.TO_FLOW,
            "max_displacement": cfg.SIMILARITY.MAX_DISPLACEMENT,
            "pad_size": cfg.SIMILARITY.PAD_SIZE,
            "flow_scale_factor": cfg.FLOW_SCALE_FACTOR,
        }

    def _corr_relu(self, features1, features2):

        corr = self.correlation_layer(features1, features2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        return self.leaky_relu(corr)

    def forward(self, feature_pyramid1, feature_pyramid2):
        """
        Performs forward pass.

        Parameters
        ----------
        feature_pyramid1 : torch.Tensor
            Input feature map of image 1

        feature_pyramid2 : torch.Tensor
            Input feature map of image 2

        Returns
        -------
        List[torch.Tensor]
            A List containing tensors of shape N x 2 x H x W representing the flow

        List[torch.Tensor]
            A List containing tensors of shape N x output_channel x H x W
        """

        up_flow, up_features = None, None
        up_flow_scale = self.flow_scale_factor * 2 ** (-(len(self.config)))

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

            if i < len(self.decoder_layers) - 1:
                up_flow = self.deconv_layers[i](flow)
                up_features = self.up_feature_layers[i](features)

        return flow_preds, features
