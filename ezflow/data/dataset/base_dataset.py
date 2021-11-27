import random

import numpy as np
import torch
import torch.utils.data as data

from ...functional import FlowAugmentor
from ...utils import read_gen


class BaseDataset(data.Dataset):
    """
    Base dataset for reading synthetic optical flow data.

    Parameters
    ----------
    is_prediction : bool, default : False
        If True,   If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    init_seed : bool, default : False
        If True, sets random seed to the worker
    augment : bool, default : False
        If True, applies data augmentation
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    aug_params : :obj:`dict`
        The parameters for data augmentation

    """

    def __init__(
        self,
        augment=True,
        aug_params={
            "crop_size": (224, 224),
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
        },
        is_prediction=False,
        init_seed=False,
    ):

        self.is_prediction = is_prediction
        self.init_seed = init_seed

        self.augmentor = None
        if augment:
            self.augmentor = FlowAugmentor(**aug_params)

        self.flow_list = []
        self.image_list = []

    def __getitem__(self, index):
        """
        Returns the corresponding images and the flow between them.

        Parameters
        ----------
        index : int
            specify the index location for access to Dataset item

        Returns
        -------
        tuple
            A tuple consisting of ((img1, img2), flow)
        """
        if self.is_prediction:
            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        flow = read_gen(self.flow_list[index])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        return (img1, img2), flow

    def __rmul__(self, v):
        """
        Returns an instance of the dataset after multiplying with v.

        """
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        """
        Return length of the dataset.

        """
        return len(self.image_list)
