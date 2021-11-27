import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class KITTI(BaseDataset):
    """
    Dataset Class for preparing the Kitti dataset for training and validation.


    """

    def __init__(
        self,
        root_dir,
        split="training",
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
        super(Kitti, self).__init__(
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
            split = "testing"

        root_dir = osp.join(root_dir, split)
        images1 = sorted(glob(osp.join(root_dir, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root_dir, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(osp.join(root_dir, "flow_occ/*_10.png")))
