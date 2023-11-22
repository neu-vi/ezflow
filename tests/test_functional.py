import numpy as np
import torch
import torchvision.transforms as transforms

from ezflow.functional import (
    FlowAugmentor,
    FlowOffsetLoss,
    MultiScaleLoss,
    Normalize,
    OffsetCrossEntropyLoss,
    SequenceLoss,
    SparseFlowAugmentor,
    crop,
)

img1 = np.random.rand(256, 256, 3).astype(np.uint8)
img2 = np.random.rand(256, 256, 3).astype(np.uint8)
flow = np.random.rand(256, 256, 2).astype(np.float32)

flow_pred = [torch.rand(4, 2, 256, 256)]
flow_gt = torch.rand(4, 2, 256, 256)

img1_tr = torch.from_numpy(img1).permute(2, 0, 1).float()
img2_tr = torch.from_numpy(img2).permute(2, 0, 1).float()


def test_crop():

    _ = crop(img1, img2, flow, crop_size=(224, 224))
    _ = crop(img1, img2, flow, crop_size=(224, 224), crop_type="random")
    _ = crop(img1, img2, flow, valid=flow, crop_size=(224, 224), sparse_transform=True)
    _ = crop(
        img1,
        img2,
        flow,
        valid=flow,
        crop_size=(224, 224),
        crop_type="random",
        sparse_transform=True,
    )


def test_FlowAugmentor():

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        noise_aug_params={"enabled": True, "aug_prob": 1.0},
        eraser_aug_params={"enabled": True, "aug_prob": 1.0},
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 1.0},
        flip_aug_params={"enabled": True, "h_flip_prob": 1.0, "v_flip_prob": 1.0},
        spatial_aug_params={"enabled": True, "aug_prob": 1.0, "stretch_prob": 1.0},
        advanced_spatial_aug_params={"enabled": False},
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        noise_aug_params={"enabled": True, "aug_prob": 1.0},
        eraser_aug_params={"enabled": True, "aug_prob": 1.0},
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 0.0},
        flip_aug_params={"enabled": True, "h_flip_prob": 1.0, "v_flip_prob": 1.0},
        spatial_aug_params={"enabled": False},
        advanced_spatial_aug_params={
            "enabled": True,
            "scale1": 0.3,
            "scale2": 0.1,
            "rotate": 0.4,
            "translate": 0.4,
            "stretch": 0.3,
            "enable_out_of_boundary_crop": False,
        },
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        noise_aug_params={"enabled": True, "aug_prob": 1.0},
        eraser_aug_params={"enabled": True, "aug_prob": 1.0},
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 0.0},
        flip_aug_params={"enabled": True, "h_flip_prob": 1.0, "v_flip_prob": 1.0},
        spatial_aug_params={"enabled": False},
        advanced_spatial_aug_params={
            "enabled": True,
            "scale1": 0.3,
            "scale2": 0.1,
            "rotate": 0.4,
            "translate": 0.4,
            "stretch": 0.3,
            "enable_out_of_boundary_crop": True,
        },
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"enabled": False},
        eraser_aug_params={"enabled": False},
        noise_aug_params={"enabled": False},
        flip_aug_params={"enabled": False},
        spatial_aug_params={"enabled": False},
        advanced_spatial_aug_params={"enabled": False},
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        noise_aug_params={"enabled": True, "aug_prob": 0.0},
        eraser_aug_params={"enabled": True, "aug_prob": 0.0},
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 0.0},
        flip_aug_params={"enabled": True, "h_flip_prob": 0.0, "v_flip_prob": 0.0},
        spatial_aug_params={"enabled": True, "aug_prob": 0.0, "stretch_prob": 0.0},
    )
    _ = augmentor(img1, img2, flow)

    del augmentor


def test_SparseFlowAugmentor():

    valid = np.random.rand(256, 256).astype(np.float32)

    augmentor = SparseFlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 1.0},
        eraser_aug_params={"enabled": True, "aug_prob": 1.0},
        spatial_aug_params={"enabled": True, "aug_prob": 1.0, "h_flip_prob": 1.0},
    )
    _ = augmentor(img1, img2, flow, valid)

    augmentor = SparseFlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"enabled": True, "asymmetric_color_aug_prob": 0.0},
        eraser_aug_params={"enabled": True, "aug_prob": 0.0},
        spatial_aug_params={"enabled": True, "aug_prob": 0.0, "h_flip_prob": 0.0},
    )
    _ = augmentor(img1, img2, flow, valid)

    del augmentor


def test_SequenceLoss():

    valid_mask = torch.randn(4, 1, 256, 256)

    loss_fn = SequenceLoss()
    _ = loss_fn(flow_pred, flow_gt, valid_mask)
    del loss_fn


def test_MultiScaleLoss():

    loss_fn = MultiScaleLoss(norm="l1")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(norm="l1", use_valid_range=True, valid_range=[[400, 400]])
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(norm="l2")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(norm="robust")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(resize_flow="upsample")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(resize_flow="downsample")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(average="mean")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    loss_fn = MultiScaleLoss(average="sum")
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn


def test_normalize():
    normalize = Normalize(use=True, mean=[0, 0, 0], std=[255.0, 255.0, 255.0])
    _ = normalize(img1_tr, img2_tr)


def test_OffsetCrossEntropyLoss():

    flow_logits = torch.randn(1, 567, 32, 32)
    offset_labs = torch.randint(0, 1, (1, 567, 32, 32))
    valid = torch.randint(0, 2, (1, 1, 256, 256))

    loss_fn = OffsetCrossEntropyLoss(weight_anneal_fn="CosineAnnealer")
    _ = loss_fn(flow_logits, offset_labs, valid, current_iter=0)
    del loss_fn, _

    params = {"power": 2}
    loss_fn = OffsetCrossEntropyLoss(weight_anneal_fn="PolyAnnealer", **params)
    _ = loss_fn(flow_logits, offset_labs, valid, current_iter=0)
    del loss_fn, _
