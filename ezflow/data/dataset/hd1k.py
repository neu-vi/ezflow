import os.path as osp
from glob import glob

import numpy as np

from .base_dataset import BaseDataset


class HD1K(BaseDataset):
    """
    Dataset Class for preparing the Kitti dataset for training and validation.


    """

    def __init__(
        self,
        root_dir,
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
        super(HD1K, self).__init__(
            augment,
            aug_params,
            is_prediction,
            init_seed,
        )

        self.is_prediction = is_prediction

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1
