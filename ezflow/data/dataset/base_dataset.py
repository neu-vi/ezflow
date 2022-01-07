import random

import numpy as np
import torch
import torch.utils.data as data

from ...utils import read_flow, read_image


class BaseDataset(data.Dataset):
    """
    Base dataset for reading synthetic optical flow data.

    Parameters
    ----------
    init_seed : bool, default : False
        If True, sets random seed to the worker
    is_prediction : bool, default : False
        If True,   If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    augment : bool, default : False
        If True, applies data augmentation
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    aug_params : :obj:`dict`
        The parameters for data augmentation

    """

    def __init__(
        self,
        init_seed=False,
        is_prediction=False,
        append_valid_mask=False,
        augment=True,
        aug_params={
            "crop_size": (224, 224),
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
        },
    ):

        self.is_prediction = is_prediction
        self.init_seed = init_seed
        self.append_valid_mask = append_valid_mask

        self.augmentor = None

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

            img1 and img2 of shape 3 x H x W.
            flow of shape 2 x H x W if append_valid_mask is False.
            flow of shape 3 x H x W if append_valid_mask is True.
        """
        if self.is_prediction:
            img1 = read_image(self.image_list[index][0])
            img2 = read_image(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]

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

        img1 = read_image(self.image_list[index][0])
        img2 = read_image(self.image_list[index][1])
        flow, valid = read_flow(self.flow_list[index])

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
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.append_valid_mask:
            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            valid = valid.float()
            valid = torch.unsqueeze(valid, dim=0)
            flow = torch.cat([flow, valid], dim=0)

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
