import torch

from openoptflow.methods import RAFT, DCVNet


def dummy_test(model, img_size=256, batch_size=2):

    img1 = torch.randn(batch_size, 3, img_size, img_size)
    img2 = torch.randn(batch_size, 3, img_size, img_size)

    flow = model(img1, img2, test_mode=True, only_flow=True)
    assert flow.shape == (batch_size, 2, img_size, img_size)


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


def test_DCVNet():

    model = DCVNet()
    dummy_test(model)
