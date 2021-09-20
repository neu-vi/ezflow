import torch

from openoptflow.models import DICL, RAFT

img1 = torch.randn(2, 3, 256, 256)
img2 = torch.randn(2, 3, 256, 256)


def test_RAFT():

    model = RAFT(small=False)
    _ = model(img1, img2)
    model.eval()
    _ = model(img1, img2, only_flow=False)
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow

    model = RAFT(small=True)
    _ = model(img1, img2)
    _ = model(img1, img2, test_mode=True)
    flow = model(img1, img2, test_mode=True, only_flow=True)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow


def test_DICL():

    model = DICL()
    _ = model(img1, img2)
    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow
