import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class FlyingChairs(BaseDataset):
    """
    Dataset Class for preparing the Flying Chair Synthetic dataset for training and validation.


    """

    def __init__(
        self,
        root_dir,
        split="training",
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
        super(FlyingChairs, self).__init__(
            augment,
            aug_params,
            is_test,
            init_seed,
        )

        images = sorted(glob(osp.join(root_dir, "*.ppm")))
        flows = sorted(glob(osp.join(root_dir, "*.flo")))
        assert len(images) // 2 == len(flows)

        if split == "validation":
            self.is_test = True

        try:
            split_list = np.loadtxt(
                osp.join(root_dir, "FlyingChairs_train_val.txt"), dtype=np.int32
            )
        except OSError:
            print("FlyingChairs_train_val.txt was not found in " + root_dir)
            exit()

        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
