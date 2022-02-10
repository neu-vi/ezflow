import torch


def endpointerror(pred, target, multi_magnitude=False):
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

    epe = torch.norm(pred - target, p=2, dim=1)

    if not multi_magnitude:
        return epe.mean().item()

    epe = epe.view(-1)
    multi_magnitude_epe = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return multi_magnitude_epe
