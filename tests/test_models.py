import torch
from torchvision import transforms as T

from ezflow.models import Predictor, build_model

img1 = torch.randn(2, 3, 256, 256)
img2 = torch.randn(2, 3, 256, 256)


def test_Predictor():

    transform = T.Compose(
        [T.Resize((224, 224)), T.ColorJitter(brightness=0.5, hue=0.3)]
    )

    predictor = Predictor("RAFT", "raft.yaml", data_transform=transform)
    flow = predictor(img1, img2)
    assert flow.shape == (2, 2, 224, 224)

    predictor = Predictor("RAFT", "raft.yaml", data_transform=transform, pad_divisor=32)
    flow = predictor(img1, img2)
    assert flow.shape == (2, 2, 224, 224)


def test_RAFT():

    model = build_model("RAFT", "raft.yaml")
    flow_preds = model(img1, img2)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)

    model.eval()
    _ = model(img1, img2, only_flow=False)
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    del model, flow, flow_preds

    _ = build_model("RAFT", default=True)


def test_DICL():

    model = build_model("DICL", "dicl.yaml")
    flow_preds = model(img1, img2)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)

    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    del model, flow, flow_preds

    _ = build_model("DICL", default=True)


def test_PWCNet():

    model = build_model("PWCNet", "pwcnet.yaml")
    flow_preds = model(img1, img2)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)

    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    del model, flow, flow_preds

    _ = build_model("PWCNet", default=True)


def test_FlowNetS():

    model = build_model("FlowNetS", "flownet_s.yaml")
    flow_preds = model(img1, img2)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)

    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    del model, flow, flow_preds

    _ = build_model("FlowNetS", default=True)


def test_FlowNetC():

    model = build_model("FlowNetC", "flownet_c.yaml")
    flow_preds = model(img1, img2)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)

    model.eval()
    flow = model(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    del model, flow, flow_preds

    _ = build_model("FlowNetC", default=True)


def test_VCN():

    model = build_model("VCN", "vcn.yaml")

    img = torch.randn(16, 3, 256, 256)

    flow_preds = model(img, img)
    assert isinstance(flow_preds, tuple) or isinstance(flow_preds, list)
    assert flow_preds[0].shape == (16, 2, 256, 256)

    model.eval()
    flow = model(img, img)
    assert flow.shape == (16, 2, 256, 256)

    del model, flow, flow_preds
