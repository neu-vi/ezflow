import socket
from contextlib import closing

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
