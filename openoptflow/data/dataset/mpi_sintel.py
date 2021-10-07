import os
import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class MpiSintel(BaseDataset):
    """
    Dataset Class for preparing the MPI Sintel Synthetic dataset for training and validation.


    """

    def __init__(
        self,
        root_dir,
        split="training",
        dstype="clean",
        is_test=False,
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
            is_test,
            init_seed,
        )
        assert (
            split == "training" or split == "test"
        ), "Incorrect split name for MPI Sintel. Accepted split values: training, test"
        image_root = osp.join(root_dir, split, dstype)
        flow_root = osp.join(root_dir, split, "flow")

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))
