import torch
from torchvision import transforms as T

from ezflow.models import Predictor, build_model

img1 = torch.randn(2, 3, 256, 256)
img2 = torch.randn(2, 3, 256, 256)


def test_Predictor():

    predictor = Predictor("RAFT", (0.0, 0.0, 0.0), (255.0, 255.0, 255.0), "raft.yaml")
    flow = predictor(img1, img2)
    assert flow.shape == (2, 2, 256, 256)

    transform = T.Compose([T.Resize((224, 224))])

    predictor = Predictor(
        "RAFT",
        (0.0, 0.0, 0.0),
        (255.0, 255.0, 255.0),
        "raft.yaml",
        data_transform=transform,
        pad_divisor=32,
    )
    flow = predictor(img1, img2)
    assert flow.shape == (2, 2, 224, 224)


def test_RAFT():

    model = build_model("RAFT", "raft.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    _ = build_model("RAFT", default=True)


def test_DICL():

    model = build_model("DICL", "dicl.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    _ = build_model("DICL", default=True)


def test_PWCNet():

    model = build_model("PWCNet", "pwcnet.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    _ = build_model("PWCNet", default=True)


def test_FlowNetS():

    model = build_model("FlowNetS", "flownet_s.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    _ = build_model("FlowNetS", default=True)


def test_FlowNetC():

    model = build_model("FlowNetC", "flownet_c.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    _ = build_model("FlowNetC", default=True)


def test_VCN():

    model = build_model("VCN", "vcn.yaml")

    img = torch.randn(16, 3, 256, 256)

    output = model(img, img)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )

    model.eval()
    output = model(img, img)
    assert output["flow_upsampled"].shape == (16, 2, 256, 256)

    del model, output


def test_DCVNet():

    model = build_model("DCVNet", "dcvnet.yaml")
    output = model(img1, img2)
    assert isinstance(output, dict)
    assert isinstance(output["flow_preds"], tuple) or isinstance(
        output["flow_preds"], list
    )
    assert isinstance(output["flow_logits"], tuple) or isinstance(
        output["flow_logits"], list
    )

    model.eval()
    output = model(img1, img2)
    assert output["flow_upsampled"].shape == (2, 2, 256, 256)

    del model, output

    model = build_model("DCVNet", default=True)
    del model
