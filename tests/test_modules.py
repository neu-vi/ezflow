import torch

from ezflow.config import CfgNode
from ezflow.modules import MODULE_REGISTRY


def test_ConvGRU():

    inp_x = torch.rand(2, 8, 32, 32)
    inp_h = torch.rand(2, 8, 32, 32)

    module = MODULE_REGISTRY.get("ConvGRU")(hidden_dim=8, input_dim=8)
    _ = module(inp_h, inp_x)


def test_BasicBlock():

    inp = torch.randn(2, 3, 256, 256)

    module = MODULE_REGISTRY.get("BasicBlock")(
        inp.shape[1], 32, norm="group", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BasicBlock")(
        inp.shape[1], 32, norm="batch", activation="leakyrelu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BasicBlock")(
        inp.shape[1], 32, norm="instance", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BasicBlock")(
        inp.shape[1], 32, norm="none", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BasicBlock")(
        inp.shape[1], 32, norm=None, activation="relu", stride=3
    )
    _ = module(inp)
    del module


def test_BottleneckBlock():

    inp = torch.randn(2, 3, 256, 256)

    module = MODULE_REGISTRY.get("BottleneckBlock")(
        inp.shape[1], 32, norm="group", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BottleneckBlock")(
        inp.shape[1], 32, norm="batch", activation="leakyrelu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BottleneckBlock")(
        inp.shape[1], 32, norm="instance", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BottleneckBlock")(
        inp.shape[1], 32, norm="none", activation="relu", stride=3
    )
    _ = module(inp)
    del module

    module = MODULE_REGISTRY.get("BottleneckBlock")(
        inp.shape[1], 32, norm=None, activation="relu", stride=3
    )
    _ = module(inp)
    del module


def test_DAP():

    inp = torch.randn(2, 1, 7, 7, 16, 16)

    module = MODULE_REGISTRY.get("DisplacementAwareProjection")(temperature=False)
    _ = module(inp)

    module = MODULE_REGISTRY.get("DisplacementAwareProjection")(temperature=True)
    _ = module(inp)


def test_ASPPConv2D():
    inp = torch.randn(2, 256, 32, 32)

    module = MODULE_REGISTRY.get("ASPPConv2D")(
        in_channels=256, hidden_dim=256, out_channels=256, norm="none"
    )
    out = module(inp)
    assert out.shape == (2, 256, 32, 32)
    del module

    module = MODULE_REGISTRY.get("ASPPConv2D")(
        in_channels=256, hidden_dim=256, out_channels=256, norm="batch"
    )
    out = module(inp)
    del module


def test_UNetBase():
    inp = torch.randn(2, 695, 32, 32)

    bottleneck_config = CfgNode(
        init_dict={
            "NAME": "ASPPConv2D",
            "IN_CHANNELS": 192,
            "HIDDEN_DIM": 192,
            "OUT_CHANNELS": 192,
            "DILATIONS": [2, 4, 8],
            "NUM_GROUPS": 1,
            "NORM": "none",
        },
        new_allowed=True,
    )

    module = MODULE_REGISTRY.get("UNetBase")(
        in_channels=695,
        hidden_dim=96,
        out_channels=96,
        bottle_neck_cfg=bottleneck_config,
    )
    out = module(inp)
    assert out.shape == (2, 96, 32, 32)
    del module


def test_UNetLight():
    inp = torch.randn(2, 695, 32, 32)

    bottleneck_config = CfgNode(
        init_dict={
            "NAME": "ASPPConv2D",
            "IN_CHANNELS": 192,
            "HIDDEN_DIM": 192,
            "OUT_CHANNELS": 192,
            "DILATIONS": [2, 4, 8],
            "NUM_GROUPS": 1,
            "NORM": "none",
        },
        new_allowed=True,
    )

    module = MODULE_REGISTRY.get("UNetLight")(
        in_channels=695,
        hidden_dim=96,
        out_channels=96,
        bottle_neck_cfg=bottleneck_config,
    )
    out = module(inp)
    assert out.shape == (2, 96, 32, 32)
    del module
