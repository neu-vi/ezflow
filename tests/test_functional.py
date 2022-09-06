import numpy as np
import torch
import torchvision.transforms as transforms

from ezflow.functional import (
    FlowAugmentor,
    MultiScaleLoss,
    Normalize,
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
        color_aug_params={"aug_prob": 1.0},
        eraser_aug_params={"aug_prob": 1.0},
        spatial_aug_params={
            "aug_prob": 1.0,
            "h_flip_prob": 1.0,
            "v_flip_prob": 1.0,
            "stretch_prob": 1.0,
        },
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"aug_prob": 0.0},
        eraser_aug_params={"aug_prob": 0.0},
        spatial_aug_params={
            "aug_prob": 0.0,
            "h_flip_prob": 0.0,
            "v_flip_prob": 0.0,
            "stretch_prob": 0.0,
        },
    )
    _ = augmentor(img1, img2, flow)

    del augmentor


def test_SparseFlowAugmentor():

    valid = np.random.rand(256, 256).astype(np.float32)

    augmentor = SparseFlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"aug_prob": 1.0},
        eraser_aug_params={"aug_prob": 1.0},
        spatial_aug_params={"aug_prob": 1.0, "h_flip_prob": 1.0},
    )
    _ = augmentor(img1, img2, flow, valid)

    augmentor = SparseFlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"aug_prob": 0.0},
        eraser_aug_params={"aug_prob": 0.0},
        spatial_aug_params={"aug_prob": 0.0, "h_flip_prob": 0.0},
    )
    _ = augmentor(img1, img2, flow, valid)

    del augmentor


def test_SequenceLoss():

    valid_mask = torch.randn(4, 1, 256, 256)
    flow_target = torch.cat([flow_gt, valid_mask], dim=1)

    loss_fn = SequenceLoss()
    _ = loss_fn(flow_pred, flow_target)
    del loss_fn


def test_MultiScaleLoss():

    loss_fn = MultiScaleLoss()
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn

    valid_mask = torch.randn(4, 1, 256, 256)
    flow_target = torch.cat([flow_gt, valid_mask], dim=1)

    loss_fn = MultiScaleLoss()
    _ = loss_fn(flow_pred, flow_target)
    del loss_fn


def test_Augmentor():
    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"aug_prob": 1.0},
        eraser_aug_params={"aug_prob": 1.0},
        spatial_aug_params={
            "aug_prob": 1.0,
            "h_flip_prob": 1.0,
            "v_flip_prob": 1.0,
            "stretch_prob": 1.0,
        },
        translate_params={
            "aug_prob": 1.0,
            "translate": 20,
        },
        rotate_params={
            "aug_prob": 1.0,
            "degrees": 20,
            "delta": 5,
        },
    )
    _ = augmentor(img1, img2, flow)

    augmentor = FlowAugmentor(
        crop_size=(224, 224),
        color_aug_params={"aug_prob": 0.0},
        eraser_aug_params={"aug_prob": 0.0},
        spatial_aug_params={
            "aug_prob": 0.0,
            "h_flip_prob": 0.0,
            "v_flip_prob": 0.0,
            "stretch_prob": 0.0,
        },
        translate_params={
            "aug_prob": 0.0,
        },
        rotate_params={
            "aug_prob": 0.0,
        },
    )
    _ = augmentor(img1, img2, flow)

    del augmentor


def test_normalize():
    normalize = Normalize(use=True, mean=[0, 0, 0], std=[255.0, 255.0, 255.0])
    _ = normalize(img1_tr, img2_tr)
