from torch.utils.data.dataloader import DataLoader

from ..dataset import *


class DataloaderCreator:
    """
    A class to configure a data loader for optical flow datasets.

    """

    def __init__(
        self, batch_size, pin_memory=False, shuffle=True, num_workers=4, drop_last=True
    ):
        self.dataset_list = []
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

    def add_flying_chairs(self, crop_size, split="training"):
        self.dataset_list.append(
            FlyingChairs(
                aug_params={
                    "crop_size": crop_size,
                    "min_scale": -0.1,
                    "max_scale": 1.0,
                    "do_flip": True,
                },
                split=split,
            )
        )

    def add_flying_things(self, split="training"):
        raise NotImplementedError

    def add_sintel(self, split="training"):
        raise NotImplementedError

    def add_kitti(self, split="training"):
        raise NotImplementedError

    def add_autoflow(self, split="training"):
        raise NotImplementedError

    def get_dataloader(self):
        assert len(self.dataset_list) == 0, "No datasets were added"

        dataset = self.dataset_list[0]

        if len(self.dataset_list > 1):
            for i in range(len(self.dataset_list) - 1):
                dataset += self.dataset_list[i + 1]

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

        print("Total image pairs: %d" % len(dataset))
        return data_loader
