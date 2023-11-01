from gettext import find

import numpy as np
import torch
import torch.nn as nn

from ezflow.utils import (
    AverageMeter,
    concentric_offsets,
    coords_grid,
    endpointerror,
    find_free_port,
    flow_to_bilinear_interpolation_weights,
    forward_interpolate,
    get_flow_offsets,
    is_port_available,
    replace_relu,
    upflow,
)


def test_endpointerror():

    pred = torch.rand(4, 2, 256, 256)
    target = torch.rand(4, 2, 256, 256)
    _ = endpointerror(pred, target)

    multi_magnitude_epe = endpointerror(pred, target, multi_magnitude=True)
    assert isinstance(multi_magnitude_epe, dict)

    target = torch.rand(
        4, 3, 256, 256
    )  # Ignore valid mask for EPE calculation if target contains it
    _ = endpointerror(pred, target)


def test_forward_interpolate():

    flow = torch.rand(2, 256, 256)
    _ = forward_interpolate(flow)


def test_upflow():

    flow = torch.rand(2, 2, 256, 256)
    _ = upflow(flow)


def test_coords_grid():

    _ = coords_grid(2, 256, 256)


def test_AverageMeter():

    meter = AverageMeter()
    meter.update(1)
    assert meter.avg == 1

    meter.reset()
    assert meter.avg == 0


def test_find_free_port():
    assert len(find_free_port()) == 5


def test_is_port_available():
    port = find_free_port()
    assert is_port_available(int(port)) is True


def test_replace_relu():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
    )

    model = replace_relu(model, nn.LeakyReLU(negative_slope=0.1))

    assert isinstance(model[1], nn.LeakyReLU)

    del model


def test_concentric_offsets():

    offset_matrix = [
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        [-36, -27, -18, -9, 0, 9, 18, 27, 36],
        [-64, -48, -32, -16, 0, 16, 32, 48, 64],
    ]

    offset_matrix = np.array(offset_matrix)
    offsets = concentric_offsets(dilations=[1, 5, 9, 16], radius=4, bias=0)

    assert offsets.shape == (4, 9)
    assert (offsets == offset_matrix).all()

    del offsets, offset_matrix


def test_get_flow_offsets():

    flow_offsets_matrix = [
        [-8, -6, -4, -2, 0, 2, 4, 6, 8],
        [-32, -24, -16, -8, 0, 8, 16, 24, 32],
        [-64, -48, -32, -16, 0, 16, 32, 48, 64],
        [-96, -72, -48, -24, 0, 24, 48, 72, 96],
        [-160, -120, -80, -40, 0, 40, 80, 120, 160],
        [-288, -216, -144, -72, 0, 72, 144, 216, 288],
        [-512, -384, -256, -128, 0, 128, 256, 384, 512],
    ]

    flow_offsets = get_flow_offsets(
        dilations=[[1], [1, 2, 3, 5, 9, 16]],
        feat_strides=[2, 8],
        radius=4,
        offset_bias=[0, 0],
        offset_fn=concentric_offsets,
    )

    assert flow_offsets.shape == (7, 9)
    assert (flow_offsets == flow_offsets_matrix).all()

    del flow_offsets, flow_offsets_matrix


def test_flow_to_bilinear_interpolation_weights():

    flow = np.ones((32, 32, 2))
    flow[:, :, 1] = flow[:, :, 1] * 2

    valid = np.ones(flow.shape[:2]) > 0

    flow_offsets = get_flow_offsets(
        dilations=[[1], [1, 2, 3]],
        feat_strides=[2, 8],
        radius=4,
        offset_bias=[0, 0],
        offset_fn=concentric_offsets,
    )

    offset_labs, dilation_labs = flow_to_bilinear_interpolation_weights(
        flow, valid, flow_offsets
    )

    assert offset_labs.shape == (32, 32, 4, 9, 9)
    assert dilation_labs.shape == (32, 32)

    offset_labs_reshape = offset_labs.reshape(32, 32, -1)
    err = np.sum(np.abs(np.sum(offset_labs_reshape, axis=2) - 1))
    assert err < 1e-10, err

    del flow, valid, offset_labs, dilation_labs
