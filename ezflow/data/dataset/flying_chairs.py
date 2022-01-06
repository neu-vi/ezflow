import os.path as osp
from glob import glob

import numpy as np

from ...functional import FlowAugmentor
from .base_dataset import BaseDataset


class FlyingChairs(BaseDataset):
    """
    Dataset Class for preparing the Flying Chair Synthetic dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the flying chairs dataset
    split : str, default : "training"
        specify the training or validation split
    is_prediction : bool, default : False
        If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    init_seed : bool, default : False
        If True, sets random seed to worker
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    augment : bool, default : True
        If True, applies data augmentation
    aug_param : :obj:`dict`, optional
        The parameters for data augmentation

    """

    def __init__(
        self,
        root_dir,
        split="training",
        is_prediction=False,
        init_seed=False,
        append_valid_mask=False,
        augment=True,
        aug_params={
            "crop_size": (224, 224),
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
        },
    ):
        super(FlyingChairs, self).__init__(
            augment, aug_params, is_prediction, init_seed, append_valid_mask
        )
        assert (
            split.lower() == "training" or split.lower() == "validation"
        ), "Incorrect split values. Accepted split values: training, validation"

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(**aug_params)

        images = sorted(glob(osp.join(root_dir, "*.ppm")))
        flows = sorted(glob(osp.join(root_dir, "*.flo")))
        assert len(images) // 2 == len(flows)

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
