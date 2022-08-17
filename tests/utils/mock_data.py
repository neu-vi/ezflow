import torch
from torch.utils.data import Dataset

from ezflow.data import DataloaderCreator


class MockOpticalFlowDataset(Dataset):
    def __init__(self, size, channels, length):

        self.length = length

        if not isinstance(size, list) and not isinstance(size, tuple):
            size = (size, size)

        self.imgs = torch.randn(length, channels, *size)
        self.flow = torch.randn(length, 2, *size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.imgs[idx], self.imgs[idx]), self.flow[idx]


class MockDataloaderCreator(DataloaderCreator):
    def __init__(self):
        super(MockDataloaderCreator, self).__init__(batch_size=1)

        self.dataset_list = []
        self.dataset_list.append(
            MockOpticalFlowDataset(size=(64, 64), channels=3, length=4)
        )
