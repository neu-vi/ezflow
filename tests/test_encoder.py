import torch

from openoptflow.encoder import BasicEncoder, BottleneckEncoder, GANetBackbone

img = torch.randn(2, 3, 256, 256)


def test_BasicEncoder():

    encoder = BasicEncoder(in_channels=3, out_channels=32)
    output = encoder(img)

    assert output.shape[:2] == (2, 32)


def test_BottleneckEncoder():

    encoder = BottleneckEncoder(in_channels=3, out_channels=32)
    output = encoder(img)

    assert output.shape[:2] == (2, 32)


def test_GANetBackbone():

    encoder = GANetBackbone(in_channels=3, out_channels=32)
    output = encoder(img)[1]

    assert output.shape[:2] == (2, 32)
