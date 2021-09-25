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
        aug_params,
        split="training",
        root_dir="datasets/FlyingChairs_release/data",
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root_dir, "*.ppm")))
        flows = sorted(glob(osp.join(root_dir, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
