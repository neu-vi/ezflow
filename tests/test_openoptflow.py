import torch

from openoptflow.methods import RAFT


def test_RAFT():

    img1 = torch.randn(2, 3, 256, 256)
    img2 = torch.randn(2, 3, 256, 256)

    model = RAFT(small=False)
    _ = model(img1, img2)
    _ = model(img1, img2, test_mode=True)
    flow = model(img1, img2, test_mode=True, only_flow=True)
    assert flow.shape == (2, 2, 256, 256)

    model = RAFT(small=True)
    _ = model(img1, img2)
    _ = model(img1, img2, test_mode=True)
    flow = model(img1, img2, test_mode=True, only_flow=True)
    assert flow.shape == (2, 2, 256, 256)
