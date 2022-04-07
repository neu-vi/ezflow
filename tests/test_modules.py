from tkinter import HIDDEN

import torch

from ezflow.modules import MODULE_REGISTRY


def test_ConvGRU():

    inp_x = torch.rand(2, 8, 32, 32)
    inp_h = torch.rand(2, 8, 32, 32)

    module = MODULE_REGISTRY.get("ConvGRU")(hidden_dim=8, input_dim=8)
    _ = module(inp_h, inp_x)
