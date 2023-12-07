import torch


def endpointerror(pred, flow_gt, valid=None, multi_magnitude=False, **kwargs):
    """
    Endpoint error

    Parameters
    ----------
    pred : torch.Tensor
        Predicted flow
    flow_gt : torch.Tensor
        flow_gt flow
    valid : torch.Tensor
        Valid flow vectors

    Returns
    -------
    torch.Tensor
        Endpoint error
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[-1]

    epe = torch.norm(pred - flow_gt, p=2, dim=1)
    f1 = None

    if valid is not None:
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid.reshape(-1) >= 0.5

        f1 = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        epe = epe[val]
        f1 = f1[val].cpu().numpy()

    if not multi_magnitude:
        if f1 is not None:
            return epe.mean().item(), f1

        return epe.mean().item()

    epe = epe.view(-1)
    multi_magnitude_epe = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    if f1 is not None:
        return multi_magnitude_epe, f1

    return multi_magnitude_epe
