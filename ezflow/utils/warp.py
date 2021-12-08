import torch
import torch.nn as nn


def warp(x, flow):
    """
    Warps an image x according to the optical flow field specified

    Parameters
    ----------
    x : torch.Tensor
        Image to be warped
    flow : torch.Tensor
        Optical flow field

    Returns
    -------
    torch.Tensor
        Warped image
    """

    B, _, H, W = x.size()

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    grid = torch.cat((xx, yy), 1).float()
    vgrid = torch.Tensor(grid).to(x.device) + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)

    output = nn.functional.grid_sample(x, vgrid, align_corners=True)

    mask = torch.ones_like(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask
