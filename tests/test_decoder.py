import torch

from ezflow.config import CfgNode
from ezflow.decoder import DECODER_REGISTRY, build_decoder
from ezflow.similarity import SIMILARITY_REGISTRY

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


def test_PyramidDecoder():
    feature_pyramid = [
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 32, 32, 32),
    ]

    decoder = DECODER_REGISTRY.get("PyramidDecoder")(config=[128, 64, 32])
    flow_preds, _ = decoder(feature_pyramid, feature_pyramid)
    for flow in flow_preds:
        assert flow.shape[1] == 2

    del decoder, feature_pyramid, flow_preds, _


def test_DCVDilatedFlowStackFilterDecoder():
    cost = torch.randn(1, 7, 9, 9, 32, 32)
    context_fmaps = [torch.randn(1, 64, 128, 128), torch.randn(1, 128, 32, 32)]

    strides = [2, 8]
    dilations = [[1], [1, 2, 3, 5, 9, 16]]

    corr_fn = SIMILARITY_REGISTRY.get("MatryoshkaDilatedCostVolumeList")(
        max_displacement=4, encoder_output_strides=strides, dilations=dilations
    )
    flow_offsets = corr_fn.get_global_flow_offsets().view(1, -1, 2, 1, 1)

    config = CfgNode(
        init_dict={
            "NAME": "DCVDilatedFlowStackFilterDecoder",
            "FEAT_STRIDES": [2, 8],
            "DILATIONS": [[1], [1, 2, 3, 5, 9, 16]],
            "COST_VOLUME_FILTER": {
                "NAME": "DCVFilterGroupConvStemJoint",
                "NUM_GROUPS": 1,
                "HIDDEN_DIM": 96,
                "FEAT_IN_PLANES": 128,
                "OUT_CHANNELS": 567,
                "SEARCH_RANGE": 9,
                "USE_FILTER_RESIDUAL": True,
                "USE_GROUP_CONV_STEM": True,
                "NORM": "none",
                "UNET": {
                    "NAME": "UNetBase",
                    "NUM_GROUPS": 1,
                    "IN_CHANNELS": 695,
                    "HIDDEN_DIM": 96,
                    "OUT_CHANNELS": 96,
                    "NORM": "none",
                    "BOTTLE_NECK": {
                        "NAME": "ASPPConv2D",
                        "IN_CHANNELS": 192,
                        "HIDDEN_DIM": 192,
                        "OUT_CHANNELS": 192,
                        "DILATIONS": [2, 4, 8],
                        "NUM_GROUPS": 1,
                        "NORM": "none",
                    },
                },
            },
        },
        new_allowed=True,
    )

    decoder = build_decoder(config)
    _ = decoder(cost, context_fmaps, flow_offsets)

    del decoder, cost, _
