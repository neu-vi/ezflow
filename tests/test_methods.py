import torch

from openoptflow.methods import DICL, RAFT


def test_RAFT():

    img1 = torch.randn(2, 3, 512, 512)
    img2 = torch.randn(2, 3, 512, 512)

    model = RAFT(small=False)
    _ = model(img1, img2)
    model = model.eval()
    _ = model(img1, img2, only_flow=False)
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 512, 512)

    model = RAFT(small=True)
    _ = model(img1, img2)
    _ = model(img1, img2, test_mode=True)
    flow = model(img1, img2, test_mode=True, only_flow=True)
    assert flow.shape == (2, 2, 512, 512)


def test_DICL():

    img1 = torch.randn(2, 3, 512, 512)
    img2 = torch.randn(2, 3, 512, 512)

    model = DICL()
    _ = model(img1, img2)
    model = model.eval()
    flow, _ = model(img1, img2)
    assert flow.shape == (2, 2, 512, 512)
