import torch

from openoptflow.decoder import DECODER_REGISTRY

cost = torch.randn(2, 81, 32, 32)


def test_ConvDecoder():

    decoder = DECODER_REGISTRY.get("ConvDecoder")(concat_channels=81)
    flow, _ = decoder(cost)
    assert flow.shape[1] == 2

    decoder = DECODER_REGISTRY.get("ConvDecoder")(config=[81, 128, 64, 32])
    flow, _ = decoder(cost)
    assert flow.shape[1] == 2

    decoder = DECODER_REGISTRY.get("ConvDecoder")(config=[81, 128, 64], to_flow=False)
    flow, _ = decoder(cost)
    assert flow.shape[1] == 64

    del decoder, flow
