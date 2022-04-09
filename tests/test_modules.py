import torch

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
