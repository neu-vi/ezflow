import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class FlyingThings3D(BaseDataset):
    """
    Dataset Class for preparing the Flying Things 3D Synthetic dataset for training and validation.


    """

    def __init__(
        self,
        root_dir,
        split="TRAIN",
        dstype="frames_cleanpass",
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
        super(FlyingThings3D, self).__init__(
            augment,
            aug_params,
            is_test,
            init_seed,
        )
        assert (
            split == "TRAIN" or split == "TEST"
        ), "Incorrect split name for Flying Things 3D. Accepted split values: TRAIN, TEST"

        if split == "TEST":
            self.is_test = True

        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                image_dirs = sorted(glob(osp.join(root_dir, dstype, split + "/*/*")))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(
                    glob(osp.join(root_dir, "optical_flow/", split + "/*/*"))
                )
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, "*.png")))
                    flows = sorted(glob(osp.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == "into_past":
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
