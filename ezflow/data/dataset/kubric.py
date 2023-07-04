import os
import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data

from ...functional import FlowAugmentor, Normalize, crop
from ...utils import read_flow, read_image
from .base_dataset import BaseDataset
from ..build import DATASET_REGISTRY
from ...config import configurable

@DATASET_REGISTRY.register()
class Kubric(BaseDataset):
    """
    Dataset Class for preparing the Kubric 'movi-f' split of
    optical flow synthetic dataset  for training and validation.
    https://arxiv.org/abs/2203.03570
    https://github.com/google-research/kubric/tree/main/challenges/optical_flow


    Note that in order to use this dataset class the Kubric Dataset
    must be in the Sintel directory structure. Please follow the script
    provided in the repository mentioned below to convert .tfrecords to
    images and flow fields arranged in the Sintel Directory structure.
    https://github.com/prajnan93/kubric-flow

    The tfrecords conversion is not provided with the ezflow package
    as it requires tensorflow installation.


    Parameters
    ----------
    root_dir : str
        path of the root directory for the MPI Sintel datasets
    split : str, default : "training"
        specify the training or validation split
    swap_column_to_row : bool, default : True
        If True, swaps column major to row major of the flow map.
        The optical flow fields were rendered in column major in the earlier versions.
        Set this parameter to False if newer versions are available in row major.
        More info in GitHub issue:https://github.com/google-research/kubric/issues/152
    use_backward_flow : bool, default : False
        returns backward optical flow field
    is_prediction : bool, default : False
        If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    init_seed : bool, default : False
        If True, sets random seed to worker
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    crop: bool, default : True
        Whether to perform cropping
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    crop_type : :obj:`str`, default : 'center'
        The type of croppping to be performed, one of "center", "random"
    augment : bool, default : True
        If True, applies data augmentation
    aug_params : :obj:`dict`, optional
        The parameters for data augmentation
    norm_params : :obj:`dict`, optional
        The parameters for normalization
    """
    @configurable
    def __init__(
        self,
        root_dir,
        split="training",
        swap_column_to_row=True,
        use_backward_flow=False,
        is_prediction=False,
        init_seed=False,
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
        norm_params={"use": False},
    ):
        super(Kubric, self).__init__(
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            sparse_transform=False,
            norm_params=norm_params,
        )

        assert (
            split.lower() == "training" or split.lower() == "validation"
        ), "Incorrect split values. Accepted split values: training, validation"

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask
        self.swap = swap_column_to_row

        if augment:
            self.augmentor = FlowAugmentor(crop_size=crop_size, **aug_params)

        split = split.lower()

        image_root = osp.join(root_dir, split, "images")

        if use_backward_flow:
            flow_root = osp.join(root_dir, split, "backward_flow")
        else:
            flow_root = osp.join(root_dir, split, "forward_flow")

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]

            if not self.is_prediction:
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

    @classmethod
    def from_config(cls, cfg):
        return {
            "root_dir": cfg.ROOT_DIR,
            "split": cfg.SPLIT,
            "swap_column_to_row": cfg.SWAP_COLUMN_TO_ROW,
            "use_backward_flow": cfg.USE_BACKWARD_FLOW,
            "is_prediction": cfg.IS_PREDICTION,
            "init_seed": cfg.INIT_SEED,
            "append_valid_mask": cfg.APPEND_VALID_MASK,
            "crop": cfg.CROP.USE,
            "crop_size": cfg.CROP.SIZE,
            "crop_type": cfg.CROP.TYPE,
            "augment": cfg.AUGMENTATION.USE,
            "aug_params": cfg.AUGMENTATION.PARAMS,
            "norm_params": cfg.NORM_PARAMS,
        }   

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

        if self.swap:
            flow_temp = np.zeros_like(flow)

            # Swap column major to row_major

            flow_temp[..., 0] = flow[..., 1]
            flow_temp[..., 1] = flow[..., 0]

            del flow
            flow = flow_temp

        if len(img1.shape) == 2:  # grayscale images
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.is_prediction:

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            img1, img2 = self.normalize(img1, img2)

            return img1, img2

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

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        img1, img2 = self.normalize(img1, img2)

        if self.append_valid_mask:
            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            valid = valid.float()
            valid = torch.unsqueeze(valid, dim=0)
            flow = torch.cat([flow, valid], dim=0)

        return (img1, img2), flow
