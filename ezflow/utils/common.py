import socket
from contextlib import closing

import numpy as np
import torch
import torch.nn as nn


def coords_grid(batch_size, h, w):
    """
    Returns a grid of coordinates in the shape of (batch_size, h, w, 2)

    Parameters
    -----------
    batch_size : int
        Batch size
    h : int
        Height of the image
    w : int
        Width of the image

    Returns
    --------
    torch.Tensor
        Grid of coordinates
    """

    coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch_size, 1, 1, 1)


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):

        self.reset()

    def reset(self):
        """
        Resets the meter
        """

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter

        Parameters
        -----------
        val : float
            Value to update the meter with
        n : int
            Number of samples to update the meter with
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_free_port():
    """
    Find an available free ports in the host machine.

    Returns
    --------
    str
        port number
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def is_port_available(port):
    """
    Check if the provided port is free in the host machine.

    Params
    ------
    port : int
        Port number of the host.

    Returns
    --------
    boolean
        Return True is the port is free otherwise False.
    """

    port = int(port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def replace_relu(module, new_fn):
    if isinstance(module, nn.DataParallel) or isinstance(
        module, nn.parallel.DistributedDataParallel
    ):
        raise Exception("Expected an nn.Module")

    assert isinstance(new_fn, (nn.LeakyReLU, nn.GELU))

    mod = module
    if isinstance(module, nn.ReLU):
        mod = new_fn

    for name, child in module.named_children():
        mod.add_module(name, replace_relu(child, new_fn))

    return mod


def concentric_offsets(dilations=[1, 5, 9, 16], radius=4, bias=0):
    offsets_list = []
    for dilation_i in dilations:
        offsets_i = np.arange(-radius, radius + 1) * dilation_i
        offsets_list.append(offsets_i.tolist())
    offsets = np.array(offsets_list)
    return offsets


def get_flow_offsets(
    dilations=[[1], [1, 2, 3, 5, 9, 16]],
    feat_strides=[2, 8],
    radius=4,
    offset_bias=[0, 0],
    offset_fn=concentric_offsets,
    **kwargs
):
    offsets_list = []
    for idx, (dilations_i, feat_stride_i) in enumerate(zip(dilations, feat_strides)):
        assert feat_stride_i <= 8
        offsets_i = offset_fn(dilations_i, radius, offset_bias[idx]) * feat_stride_i
        offsets_list.append(offsets_i)
    offsets = np.concatenate(offsets_list, axis=0)

    return offsets


def flow_to_bilinear_interpolation_weights_helper(flow_x, offsets, debug=False):
    h, w = flow_x.shape

    # HW * D * O
    diff_x = flow_x.reshape(h * w, 1, 1) - offsets[None]

    # check where the two candidates cross the position of interest
    flag = np.logical_xor(diff_x[:, :, :-1] > 0, diff_x[:, :, 1:] > 0)

    # regular start point
    flag_start = np.logical_and(flag, diff_x[:, :, :-1] > 0)

    # check the 0 start point
    flag_start[:, :, 0] = np.logical_or(flag_start[:, :, 0], diff_x[:, :, 0] == 0)

    # append all false values to the last column as they can not be the start point
    flag_start = np.concatenate(
        (flag_start, np.zeros((flag_start.shape[0], flag_start.shape[1], 1)) > 0),
        axis=2,
    )
    if not np.all(np.sum(flag_start, axis=2) <= 1) and debug:
        import pdb

        pdb.set_trace()
        debug_data = {"flow_x": flow_x, "offsets": offsets}
        torch.save(debug_data, "debug_offset_labs.pkl")
    assert np.all(np.sum(flag_start, axis=2) <= 1)

    # get the end point by shifting the start point one by right
    flag_end = np.concatenate(
        (
            np.zeros((flag_start.shape[0], flag_start.shape[1], 1)) > 0,
            flag_start[:, :, :-1],
        ),
        axis=2,
    )

    idxes_start = np.argmax(flag_start, axis=2)
    idxes_end = np.argmax(flag_end, axis=2)
    diff_x_start = np.take_along_axis(diff_x, idxes_start[:, :, None], axis=2)
    diff_x_end = np.take_along_axis(diff_x, idxes_end[:, :, None], axis=2)
    min_diff_x = np.minimum(np.abs(diff_x_start), np.abs(diff_x_end))

    # set the out-of-range positions to be inf
    out_of_range_idxes = np.sum(flag_start, axis=2) == 0
    min_diff_x = min_diff_x + out_of_range_idxes[:, :, None].astype(np.float32) * 100000

    return diff_x, idxes_start, idxes_end, min_diff_x


def get_bilinear_weights_per_pixel(x, y, x1, y1, x2, y2):
    w11 = (x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-30)
    w12 = (x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-30)
    w21 = (x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-30)
    w22 = (x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-30)
    return w11, w12, w21, w22


def flow_to_bilinear_interpolation_weights(flow, valid, flow_offsets, debug=False):
    """
    Get the bilinear interpolation weights using flow_offsets.

    Input:
    - flow: HxWx2, numpy array
    - valid: HxW, valid flag for each pixel
    - flow_offsets: DxO, D is number of dilations, O is number of offsets. It is assumed that:
      i)  the offsets in x and y directions are the same.
      ii) the offsets for each dilation is arranged in an ascending order.

    Output:
    - offsets_labs: bilinear interpolation weights for each pixel, which sum to 1. HxWxDxO^2
    - dilation_labs: in which dilation factor, are the bilinear interpolation weights found. HxWxD
    """
    if valid is None:
        valid = np.ones(flow.shape[:2]) > 0

    invalid = np.logical_not(valid)
    max_flow = flow_offsets.max()
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    u[invalid] = max_flow
    v[invalid] = max_flow
    flow = np.stack((u, v), axis=2)
    assert np.all(np.diff(flow_offsets, axis=1) > 0)

    (
        diff_x,
        x_idxes_start,
        x_idxes_end,
        min_diff_x,
    ) = flow_to_bilinear_interpolation_weights_helper(
        flow[:, :, 1], flow_offsets, debug
    )

    (
        diff_y,
        y_idxes_start,
        y_idxes_end,
        min_diff_y,
    ) = flow_to_bilinear_interpolation_weights_helper(
        flow[:, :, 0], flow_offsets, debug
    )

    min_diff = min_diff_x + min_diff_y
    min_dilation_idxes = np.argmin(min_diff, axis=1).squeeze()

    h, w, c = flow.shape
    flow = flow.reshape(h * w, c)
    num_dilation, num_disp = flow_offsets.shape
    offset_labs = np.zeros((h * w, num_dilation, num_disp, num_disp))

    # for each pixel
    for idx in range(h * w):
        best_dilation_idx = min_dilation_idxes[idx]
        x_idx_start = x_idxes_start[idx, best_dilation_idx]
        x_idx_end = x_idxes_end[idx, best_dilation_idx]
        y_idx_start = y_idxes_start[idx, best_dilation_idx]
        y_idx_end = y_idxes_end[idx, best_dilation_idx]
        x_disp_start = flow_offsets[best_dilation_idx, x_idx_start]
        x_disp_end = flow_offsets[best_dilation_idx, x_idx_end]
        y_disp_start = flow_offsets[best_dilation_idx, y_idx_start]
        y_disp_end = flow_offsets[best_dilation_idx, y_idx_end]

        w11, w12, w21, w22 = get_bilinear_weights_per_pixel(
            flow[idx, 1],
            flow[idx, 0],
            x_disp_start,
            y_disp_start,
            x_disp_end,
            y_disp_end,
        )

        if np.abs(w11 + w12 + w21 + w22 - 1) > 1e-10 and debug:
            import pdb

            pdb.set_trace()
            debug_data = {
                "flow": flow.reshape(h, w, -1),
                "offsets": flow_offsets,
                "valid": valid,
                "idx": idx,
                "diff_x": diff_x,
                "x_idxes_start": x_idxes_start,
                "x_idxes_end": x_idxes_end,
                "min_diff_x": min_diff_x,
                "diff_y": diff_y,
                "y_idxes_start": y_idxes_start,
                "y_idxes_end": y_idxes_end,
                "min_diff_y": min_diff_y,
            }
            torch.save(debug_data, "debug_data_w.pkl")

        assert np.abs(w11 + w12 + w21 + w22 - 1) < 1e-10
        offset_labs[idx, best_dilation_idx, x_idx_start, y_idx_start] = w11
        offset_labs[idx, best_dilation_idx, x_idx_start, y_idx_end] = w12
        offset_labs[idx, best_dilation_idx, x_idx_end, y_idx_start] = w21
        offset_labs[idx, best_dilation_idx, x_idx_end, y_idx_end] = w22

    offset_labs = offset_labs.reshape(h, w, num_dilation, num_disp, num_disp)
    dilation_labs = min_dilation_idxes.reshape(h, w)

    # sanity check
    offset_labs_reshape = offset_labs.reshape(h, w, -1)
    err = np.sum(np.abs(np.sum(offset_labs_reshape, axis=2) - 1))
    assert err < 1e-10, err

    return offset_labs, dilation_labs
