def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device)


class DeviceDataLoader:
    """
    A data loader wrapper to move data to a specific compute device.

    Parameters
    ----------
    data_loader : DataLoader
        The PyTorch DataLoader from torch.utils.data.dataloader
    device : torch.device
        The compute device
    """

    def __init__(self, data_loader, device):

        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        """
        Yield a batch of data after moving it to a device.

        """
        for batch in self.data_loader:
            yield to_device(batch, self.device)

    def __len__(self):
        """
        Return the number of batches.

        """
        return len(self.data_loader)
