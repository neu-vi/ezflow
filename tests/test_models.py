import torch
from torchvision import transforms as T

from openoptflow.models import DefaultPredictor, build_model

img1 = torch.randn(2, 3, 256, 256)
img2 = torch.randn(2, 3, 256, 256)


def test_DefaultPredictor():

    transform = T.Compose(
        [T.Resize((224, 224)), T.ColorJitter(brightness=0.5, hue=0.3)]
    )

    predictor = DefaultPredictor("RAFT", "raft.yaml", data_transform=transform)
    flow = predictor(img1, img2)
    assert flow.shape == (2, 2, 224, 224)


def test_RAFT():

    model = build_model("RAFT", "raft.yaml")
    _ = model(img1, img2)
    model.eval()
    _ = model(img1, img2, only_flow=False)
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow

    _ = build_model("RAFT", default=True)


def test_DICL():

    model = build_model("DICL", default=True)
    _ = model(img1, img2)
    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow


def test_PWCNet():

    model = build_model("PWCNet", default=True)
    _ = model(img1, img2)
    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)
    del model, flow
