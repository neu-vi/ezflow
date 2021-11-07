import torch

from openoptflow.decoder import DECODER_REGISTRY

cost = torch.randn(2, 81, 32, 32)


def test_ConvDecoder():

    decoder = DECODER_REGISTRY.get("ConvDecoder")(concat_channels=81)
    output = decoder(cost)
    assert output.shape[1] == 2

    decoder = DECODER_REGISTRY.get("ConvDecoder")(config=[81, 128, 64, 32])
    output = decoder(cost)
    assert output.shape[1] == 2

    decoder = DECODER_REGISTRY.get("ConvDecoder")(config=[81, 128, 64], to_flow=False)
    output = decoder(cost)
    assert output.shape[1] == 64

    del decoder, output
