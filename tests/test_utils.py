import torch

from ezflow.utils import endpointerror


def test_endpointerror():

    pred = torch.rand(4, 2, 256, 256)
    target = torch.rand(4, 2, 256, 256)
    _ = endpointerror(pred, target)

    target = torch.rand(
        4, 3, 256, 256
    )  # Ignore valid mask for EPE calculation if target contains it
    _ = endpointerror(pred, target)
