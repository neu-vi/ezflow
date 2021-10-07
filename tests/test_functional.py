import numpy as np
import torch

from openoptflow.functional import FlowAugmentor, MultiScaleLoss, SequenceLoss

img1 = np.random.rand(256, 256, 3).astype(np.uint8)
img2 = np.random.rand(256, 256, 3).astype(np.uint8)
flow = np.random.rand(256, 256, 2).astype(np.float32)

flow_pred = [torch.rand(4, 2, 256, 256)]
flow_gt = torch.rand(4, 2, 256, 256)


def test_FlowAugmentor():

    augmentor = FlowAugmentor((224, 224))
    _ = augmentor(img1, img2, flow)


def test_SequenceLoss():

    loss_fn = SequenceLoss()
    _ = loss_fn(flow_pred, flow_gt)


def test_MultiScaleLoss():

    loss_fn = MultiScaleLoss()
    _ = loss_fn(flow_pred, flow_gt)
