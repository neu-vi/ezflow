import torch

from openoptflow.methods import RAFT


def test_RAFT():

    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)

    model = RAFT()
    flow = model(img1, img2)

    assert flow.shape == (1, 2, 256, 256)
