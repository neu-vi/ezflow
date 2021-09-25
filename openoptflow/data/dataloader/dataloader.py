from ..dataset import *


class DataloaderCreator:
    """
    A class to configure and fetch data loaders for optical flow datasets.

    """

    def __init__(self):
        self.datasets = []

    def add_flying_chairs(self):
        raise NotImplementedError

    def add_flying_things(self):
        raise NotImplementedError

    def add_sintel(self):
        raise NotImplementedError

    def add_kitti(self):
        raise NotImplementedError

    def add_autoflow(self):
        raise NotImplementedError

    def fetch_dataloader(self):
        raise NotImplementedError
