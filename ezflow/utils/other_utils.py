import torch


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

    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
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
