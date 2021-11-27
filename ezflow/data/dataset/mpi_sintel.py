import os
import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class MPISintel(BaseDataset):
    """
    Dataset Class for preparing the MPI Sintel Synthetic dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the MPI Sintel datasets
    split : str, default : "training"
        specify the training or validation split
    dstype : str, default : "frames_cleanpass"
        specify dataset type
    is_prediction : bool, default : False
        If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    init_seed : bool, default : False
        If True, sets random seed to worker
    augment : bool, default : True
        If True, applies data augmentation
    aug_param : :obj:`dict`, optional
        The parameters for data augmentation
    """

    def __init__(
        self,
        root_dir,
        split="training",
        dstype="clean",
        is_prediction=False,
        init_seed=False,
        augment=True,
        aug_params={
            "crop_size": (224, 224),
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
        },
    ):
        super(MpiSintel, self).__init__(
            augment,
            aug_params,
            is_prediction,
            init_seed,
        )
        assert (
            split.lower() == "training" or split.lower() == "validation"
        ), "Incorrect split values. Accepted split values: training, validation"

        self.is_prediction = is_prediction

        split = split.lower()
        if split == "validation":
            split = "test"

        image_root = osp.join(root_dir, split, dstype)
        flow_root = osp.join(root_dir, split, "flow")

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))
