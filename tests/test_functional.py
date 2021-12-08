import numpy as np
import torch

from ezflow.functional import FlowAugmentor, MultiScaleLoss, SequenceLoss

img1 = np.random.rand(256, 256, 3).astype(np.uint8)
img2 = np.random.rand(256, 256, 3).astype(np.uint8)
flow = np.random.rand(256, 256, 2).astype(np.float32)

flow_pred = [torch.rand(4, 2, 256, 256)]
flow_gt = torch.rand(4, 2, 256, 256)


def test_FlowAugmentor():

    augmentor = FlowAugmentor(
        (224, 224),
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
        (224, 224),
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


def test_SequenceLoss():

    loss_fn = SequenceLoss()
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn


def test_MultiScaleLoss():

    loss_fn = MultiScaleLoss()
    _ = loss_fn(flow_pred, flow_gt)
    del loss_fn
