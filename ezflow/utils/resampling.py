import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate


def forward_interpolate(flow):
    """
    Forward interpolation of flow field

    Parameters
    ----------
    flow : torch.Tensor
        Flow field to be interpolated

    Returns
    -------
    torch.Tensor
        Forward interpolated flow field

    """

    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)

    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mask=False):
    """
    Biliear sampler for images

    Parameters
    ----------
    img : torch.Tensor
        Image to be sampled
    coords : torch.Tensor
        Coordinates to be sampled

    Returns
    -------
    torch.Tensor
        Sampled image
    """

    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def upflow(flow, scale=8, mode="bilinear"):
    """
    Interpolate flow field

    Parameters
    ----------
    flow : torch.Tensor
        Flow field to be interpolated
    scale : int
        Scale of the interpolated flow field
    mode : str
        Interpolation mode

    Returns
    -------
    torch.Tensor
        Interpolated flow field
    """

    new_size = (scale * flow.shape[2], scale * flow.shape[3])

    return scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def convex_upsample_flow(flow, mask_logits, out_stride):  # adapted from RAFT
    """
    Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination

    Parameters
    ----------
    flow : torch.Tensor
        Flow field to be upsampled
    mask_logits : torch.Tensor
        Mask logits
    out_stride : int
        Output stride

    Returns
    -------
    torch.Tensor
        Upsampled flow field
    """

    N, C, H, W = flow.shape
    mask_logits = mask_logits.view(N, 1, 9, out_stride, out_stride, H, W)
    mask_probs = torch.softmax(mask_logits, dim=2)

    up_flow = F.unfold(flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask_probs * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(N, C, out_stride * H, out_stride * W)
