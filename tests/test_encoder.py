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


def test_ConvEncoder():

    encoder = ENCODER_REGISTRY.get("ConvEncoder")(
        in_channels=3, channels=(16, 32, 64), kernels=(3, 3, 3), strides=(1, 1, 1)
    )
    outputs = encoder(img)

    assert len(outputs) == 3, "Number of outputs do not match"
    assert outputs[0].shape[:2] == (2, 16), "Number of output channels do not match"
    assert outputs[1].shape[:2] == (2, 32), "Number of output channels do not match"
    assert outputs[2].shape[:2] == (2, 64), "Number of output channels do not match"

    del encoder

    encoder = ENCODER_REGISTRY.get("ConvEncoder")(
        in_channels=3,
        channels=(16, 32, 32, 64, 64),
        kernels=(3, 3, 3, 3, 3),
        strides=(1, 1, 1, 1, 1),
    )
    outputs = encoder(img)

    assert len(outputs) == 3, "Number of outputs do not match"
    assert outputs[0].shape[:2] == (2, 16), "Number of output channels do not match"
    assert outputs[1].shape[:2] == (2, 32), "Number of output channels do not match"
    assert outputs[2].shape[:2] == (2, 64), "Number of output channels do not match"

    del encoder
