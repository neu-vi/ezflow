import torch


def endpointerror(pred, target, multi_magnitude=False, valid_range=None):
    """
    Endpoint error

    Parameters
    ----------
    pred : torch.Tensor
        Predicted flow
    target : torch.Tensor
        Target flow
    multi_magnitude : bool, default=False
        If True, computes epe for the magnitudes 1px, 3px and 5px
    valid_range : tuple, default=None
        range for the valid flow mask

    Returns
    -------
    torch.Tensor
        Endpoint error
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[-1]

    extra_mask = None
    if target.shape[1] == 3:
        """Ignore valid mask for EPE calculation."""
        extra_mask = target[:, 2:, :, :]
        extra_mask = torch.squeeze(extra_mask, dim=1)

        target = target[:, :2, :, :]

    if valid_range is not None:
        mask = (target[:, 0, :, :].abs() <= valid_range[1]) & (
            target[:, 1, :, :].abs() <= valid_range[0]
        )
        mask = mask.unsqueeze(1).expand(-1, 2, -1, -1).float()

        pred = pred * mask
        target = target * mask

    epe = torch.norm(pred - target, p=2, dim=1)

    if extra_mask is not None:
        epe = epe[extra_mask.byte()]

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
