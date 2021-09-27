import torch


def endpointerror(pred, target):
    """
    Endpoint error.
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[0]

    epe = torch.norm(target - pred, p=2, dim=1).mean()

    return epe
