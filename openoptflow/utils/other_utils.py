import torch


def coords_grid(batch_size, h, w):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)
