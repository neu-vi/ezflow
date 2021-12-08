import torch

from ezflow.decoder import DECODER_REGISTRY

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

    decoder = DECODER_REGISTRY.get("FlowNetConvDecoder")()
    flow_preds = decoder(
        [
            torch.randn(2, 64, 128, 128),
            torch.randn(2, 128, 64, 64),
            torch.randn(2, 256, 32, 32),
            torch.randn(2, 512, 16, 16),
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 1024, 4, 4),
        ]
    )
    assert type(flow_preds) == list and len(flow_preds) == 5
    assert flow_preds[0].shape[1] == 2

    del decoder, flow


def test_SeparableConv4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    decoder = DECODER_REGISTRY.get("SeparableConv4D")(2, 4)
    _ = decoder(inp)

    del decoder, inp


def test_Butterfly4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    decoder = DECODER_REGISTRY.get("Butterfly4D")(2, 4)
    _ = decoder(inp)

    del decoder, inp


def test_SoftArg2DFlowRegression():

    decoder = DECODER_REGISTRY.get("SoftArg2DFlowRegression")()
    cost = torch.randn([2, 1, 7, 7, 4, 4])
    flow = decoder(cost)
    assert flow.shape[1] == 2

    del decoder, flow


def test_Soft4DFlowRegression():

    decoder = DECODER_REGISTRY.get("Soft4DFlowRegression")(
        size=(2, 4, 4), max_disp=2, entropy=True, factorization=1
    )
    cost = torch.randn(2, 5, 5, 4, 4)
    flow, entropy = decoder(cost)
    assert flow.shape[1] == 2
    assert entropy.shape[1] == 2

    decoder = DECODER_REGISTRY.get("Soft4DFlowRegression")(
        size=(2, 4, 4), max_disp=2, entropy=False, factorization=1
    )
    cost = torch.randn(2, 5, 5, 4, 4)
    flow = decoder(cost)
    assert flow[0].shape[1] == 2

    del decoder, flow
