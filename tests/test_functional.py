import torch

from openoptflow.functional import MultiScaleLoss, SequenceLoss

flow_pred = torch.rand(2, 2, 256, 256)
flow_gt = torch.rand(2, 2, 256, 256)


def test_SequenceLoss():

    loss_fn = SequenceLoss()
    _ = loss_fn(flow_pred, flow_gt)


def test_MultiScaleLoss():

    loss_fn = MultiScaleLoss()
    _ = loss_fn(flow_pred, flow_gt)
