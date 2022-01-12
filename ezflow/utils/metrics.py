import torch


def endpointerror(pred, target):
    """
    Endpoint error

    Parameters
    ----------
    pred : torch.Tensor
        Predicted flow
    target : torch.Tensor
        Target flow

    Returns
    -------
    torch.Tensor
        Endpoint error
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[-1]

    if target.shape[1] == 3:
        """Ignore valid mask for EPE calculation."""
        target = target[:, :2, :, :]

    epe = torch.norm(target - pred, p=2, dim=1).mean()

    return epe
