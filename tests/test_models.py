import torch

from openoptflow.models import DICL, RAFT, build_model

img1 = torch.randn(2, 3, 256, 256)
img2 = torch.randn(2, 3, 256, 256)


def test_RAFT():

    model = build_model("RAFT", "raft.yaml")
    _ = model(img1, img2)
    model.eval()
    _ = model(img1, img2, only_flow=False)
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow

    model = build_model("RAFT", default=True)


def test_DICL():

    model = DICL()
    _ = model(img1, img2)
    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow
