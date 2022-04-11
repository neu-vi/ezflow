from gettext import find

import torch

from ezflow.utils import (
    AverageMeter,
    coords_grid,
    endpointerror,
    find_free_port,
    forward_interpolate,
    is_port_available,
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
