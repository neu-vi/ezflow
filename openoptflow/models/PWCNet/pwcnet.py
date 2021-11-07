import torch
import torch.nn as nn
import torch.nn.functional as F

from ...decoder import build_decoder
from ...encoder import build_encoder
from ...functional import CorrelationLayer
from ...utils import warp
from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PWCNet(nn.Module):
    def __init__(self, cfg):
        super(PWCNet, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)
        self.correlation_layer = CorrelationLayer(
            pad_size=cfg.CORRELATION.PAD_SIZE,
            max_displacement=cfg.CORRELATION.MAX_DISPLACEMENT,
        )

        self.decoder = build_decoder(cfg.DECODER)
