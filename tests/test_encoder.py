import torch

from openoptflow.encoder import ENCODER_REGISTRY

img = torch.randn(2, 3, 256, 256)


def test_BasicEncoder():

    encoder = ENCODER_REGISTRY.get("BasicEncoder")(in_channels=3, out_channels=32)
    output = encoder(img)
    assert output.shape[:2] == (2, 32)

    del encoder, output


def test_BottleneckEncoder():

    encoder = ENCODER_REGISTRY.get("BottleneckEncoder")(in_channels=3, out_channels=32)
    output = encoder(img)
    assert output.shape[:2] == (2, 32)

    del encoder, output


def test_GANetBackbone():

    encoder = ENCODER_REGISTRY.get("GANetBackbone")(in_channels=3, out_channels=32)
    output = encoder(img)[1]
    assert output.shape[:2] == (2, 32)

    del encoder, output


def test_PyramidEncoder():

    encoder = ENCODER_REGISTRY.get("PyramidEncoder")(in_channels=3, config=(16, 32, 64))
    feature_pyramid = encoder(img)
    assert len(feature_pyramid) == 3

    del encoder, feature_pyramid
