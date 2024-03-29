import torch

from ezflow.encoder import ENCODER_REGISTRY

img = torch.randn(2, 3, 256, 256)


def test_BasicEncoder():

    encoder_class = ENCODER_REGISTRY.get("BasicEncoder")

    encoder = encoder_class(in_channels=3, layer_config=(32, 64))
    output = encoder(img)
    assert output.shape[:2] == (2, 64)

    encoder = encoder_class(
        in_channels=3,
        layer_config=(32, 64, 96),
        intermediate_features=True,
    )
    output = encoder(img)
    assert isinstance(output, list) and len(output) == 3

    encoder = encoder_class(
        in_channels=3, layer_config=(32, 64), norm="none", p_dropout=0.1
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 64)

    encoder = encoder_class(
        in_channels=3, layer_config=(32, 64), norm="batch", p_dropout=0.1
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 64)

    encoder = encoder_class(
        in_channels=3, layer_config=(32, 64), norm="instance", p_dropout=0.1
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 64)

    encoder = encoder_class(
        in_channels=3, layer_config=(32, 64), norm="group", p_dropout=0.1
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 64)

    del encoder, output


def test_BottleneckEncoder():

    encoder_class = ENCODER_REGISTRY.get("BottleneckEncoder")

    encoder = encoder_class(in_channels=3, layer_config=(32, 64, 96))
    output = encoder(img)
    assert output.shape[:2] == (2, 96)

    encoder = encoder_class(
        in_channels=3,
        layer_config=(32, 64, 96),
        intermediate_features=True,
    )
    output = encoder(img)
    assert isinstance(output, list) and len(output) == 3

    encoder = encoder_class(
        in_channels=3, layer_config=(32, 64, 96), norm="group", p_dropout=0.1
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 96)

    encoder = encoder_class(in_channels=3, layer_config=(32, 64, 96), norm="instance")
    output = encoder(img)
    assert output.shape[:2] == (2, 96)

    encoder = encoder_class(in_channels=3, layer_config=(32, 64, 96), norm="none")
    output = encoder(img)
    assert output.shape[:2] == (2, 96)

    del encoder, output


def test_RAFTBackbone():
    encoder_class = ENCODER_REGISTRY.get("RAFTBackbone")

    encoder = encoder_class(
        in_channels=3, out_channels=256, norm="instance", layer_config=(64, 96, 128)
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 256)


def test_RAFTBackboneSmall():
    encoder_class = ENCODER_REGISTRY.get("RAFTBackboneSmall")

    encoder = encoder_class(
        in_channels=3, out_channels=128, norm="instance", layer_config=(32, 64, 96)
    )
    output = encoder(img)
    assert output.shape[:2] == (2, 128)


def test_DCVNetBackbone():
    encoder_class = ENCODER_REGISTRY.get("DCVNetBackbone")

    encoder = encoder_class(
        in_channels=3, out_channels=256, norm="instance", layer_config=(64, 96, 128)
    )
    feat, context = encoder(img)

    assert context[0].shape[:2] == (2, 64)
    assert context[1].shape[:2] == (2, 128)
    assert feat[0].shape[:2] == (2, 128)
    assert feat[1].shape[:2] == (2, 256)


def test_GANetBackbone():

    encoder = ENCODER_REGISTRY.get("GANetBackbone")(in_channels=3, out_channels=32)
    output = encoder(img)[1]
    assert output.shape[:2] == (2, 32)

    del encoder, output


def test_PyramidEncoder():

    encoder = ENCODER_REGISTRY.get("PyramidEncoder")(in_channels=3, config=(16, 32, 64))
    feature_pyramid = encoder(img)
    assert isinstance(feature_pyramid, list) or isinstance(feature_pyramid, tuple)
    assert len(feature_pyramid) == 3

    del encoder, feature_pyramid


def test_PSPNetBackbone():

    encoder_class = ENCODER_REGISTRY.get("PSPNetBackbone")

    encoder = encoder_class()
    feature_pyramid = encoder(img)
    assert isinstance(feature_pyramid, list) or isinstance(feature_pyramid, tuple)
    assert len(feature_pyramid) == 5  # PSPNetBackbone returns 5 feature maps

    encoder = encoder_class(norm=False)
    feature_pyramid = encoder(img)
    assert isinstance(feature_pyramid, list) or isinstance(feature_pyramid, tuple)
    assert len(feature_pyramid) == 5  # PSPNetBackbone returns 5 feature maps

    del encoder, feature_pyramid


def test_BasicConvEncoder():

    encoder = ENCODER_REGISTRY.get("BasicConvEncoder")(
        in_channels=3, config=(16, 32, 64)
    )
    outputs = encoder(img)

    assert len(outputs) == 3, "Number of outputs do not match"
    assert outputs[0].shape[:2] == (2, 16), "Number of output channels do not match"
    assert outputs[1].shape[:2] == (2, 32), "Number of output channels do not match"
    assert outputs[2].shape[:2] == (2, 64), "Number of output channels do not match"

    del encoder


def test_FlownetConvEncoder():

    encoder_class = ENCODER_REGISTRY.get("FlowNetConvEncoder")

    encoder = encoder_class(in_channels=3, config=(16, 32, 64))
    outputs = encoder(img)

    assert len(outputs) == 3, "Number of outputs do not match"
    assert outputs[0].shape[:2] == (2, 16), "Number of output channels do not match"
    assert outputs[1].shape[:2] == (2, 32), "Number of output channels do not match"
    assert outputs[2].shape[:2] == (2, 64), "Number of output channels do not match"

    del encoder

    encoder = encoder_class(in_channels=3, config=(16, 32, 64, 64))
    outputs = encoder(img)

    assert len(outputs) == 3, "Number of outputs do not match"
    assert outputs[0].shape[:2] == (2, 16), "Number of output channels do not match"
    assert outputs[1].shape[:2] == (2, 32), "Number of output channels do not match"
    assert outputs[2].shape[:2] == (2, 64), "Number of output channels do not match"

    del encoder
