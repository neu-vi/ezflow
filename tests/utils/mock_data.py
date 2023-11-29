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
        self.valid = torch.ones(1, *size)
        self.offset_labs = torch.randint(0, 1, (1, 567, 32, 32))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target = {}
        target["flow_gt"] = self.flow[idx]
        target["valid"] = self.valid
        target["offset_labs"] = self.offset_labs
        return (self.imgs[idx], self.imgs[idx]), target


class MockDataloaderCreator(DataloaderCreator):
    def __init__(self):
        super(MockDataloaderCreator, self).__init__(batch_size=1)

        self.dataset_list = []
        self.dataset_list.append(
            MockOpticalFlowDataset(size=(64, 64), channels=3, length=4)
        )
