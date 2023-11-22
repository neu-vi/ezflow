import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from ...functional import Normalize, crop
from ...utils import (
    flow_to_bilinear_interpolation_weights,
    get_flow_offsets,
    read_flow,
    read_image,
)


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
    crop: bool, default : True
        Whether to perform cropping
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    crop_type : :obj:`str`, default : 'center'
        The type of croppping to be performed, one of "center", "random"
    augment : bool, default : False
        If True, applies data augmentation
    aug_params : :obj:`dict`
        The parameters for data augmentation
    norm_params : :obj:`dict`, optional
        The parameters for normalization
    flow_offset_params: :obj:`dict`, optional
        The parameters for adding bilinear interpolated weights surrounding each ground truth flow values.
    """

    def __init__(
        self,
        init_seed=False,
        is_prediction=False,
        append_valid_mask=False,
        crop=False,
        crop_size=(256, 256),
        crop_type="center",
        augment=True,
        aug_params={
            "eraser_aug_params": {"enabled": False},
            "noise_aug_params": {"enabled": False},
            "flip_aug_params": {"enabled": False},
            "color_aug_params": {"enabled": False},
            "spatial_aug_params": {"enabled": False},
            "advanced_spatial_aug_params": {"enabled": False},
        },
        sparse_transform=False,
        norm_params={"use": False},
        flow_offset_params={
            "use": False,
            "dilations": [[1], [1, 2, 3, 5, 9, 16]],
            "feat_strides": [2, 8],
            "search_radius": 4,
            "offset_bias": [0, 0],
        },
    ):

        self.is_prediction = is_prediction
        self.init_seed = init_seed
        self.append_valid_mask = append_valid_mask
        self.crop = crop
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.sparse_transform = sparse_transform

        self.augment = augment
        self.augmentor = None

        self.flow_list = []
        self.image_list = []
        self.normalize = Normalize(**norm_params)

        self.flow_offsets = None
        if flow_offset_params["use"]:
            self.flow_offsets = get_flow_offsets(**flow_offset_params)

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
            A tuple consisting of ((img1, img2), dict)

            img1 and img2 of shape 3 x H x W.
            dictionary containing flow of shape 2 x H x W, valid mask of shape 1 x H x W
        """

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

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if len(img1.shape) == 2:  # grayscale images
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.is_prediction:
            if self.crop:
                img1, img2, _, _ = crop(
                    img1,
                    img2,
                    flow=None,
                    valid=None,
                    crop_size=self.crop_size,
                    crop_type=self.crop_type,
                    sparse_transform=False,
                )

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            img1, img2 = self.normalize(img1, img2)
            return img1, img2

        flow, valid = read_flow(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)

        if self.augment is True and self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        if self.crop is True:
            img1, img2, flow, valid = crop(
                img1,
                img2,
                flow,
                valid=valid,
                crop_size=self.crop_size,
                crop_type=self.crop_type,
                sparse_transform=self.sparse_transform,
            )

        if self.flow_offsets is not None:
            offset_labs = self._flow_to_bilinear_interpolation_weights(flow, valid)
            offset_labs = torch.from_numpy(offset_labs).float()
            offset_labs = offset_labs.view(
                offset_labs.shape[0], offset_labs.shape[1], -1
            ).permute(2, 0, 1)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        img1, img2 = self.normalize(img1, img2)
        target = {}
        target["flow_gt"] = flow

        if self.append_valid_mask:
            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            valid = valid.float()
            valid = torch.unsqueeze(valid, dim=0)
            target["valid"] = valid

        if self.flow_offsets is not None:
            target["offset_labs"] = offset_labs

        return (img1, img2), target

    def _flow_to_bilinear_interpolation_weights(self, flow, valid):
        max_flow = np.max(self.flow_offsets)
        valid_offsets = np.logical_and(
            np.abs(flow[:, :, 0]) <= max_flow, np.abs(flow[:, :, 1]) <= max_flow
        )
        if valid is None:
            valid = valid_offsets
        else:
            valid = np.logical_and(valid, valid_offsets)

        flow_downsample = flow[::8, ::8]
        offset_labs, dilation_labs = flow_to_bilinear_interpolation_weights(
            flow_downsample, valid[::8, ::8], self.flow_offsets
        )
        return offset_labs

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
