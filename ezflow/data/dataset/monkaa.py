import os
from glob import glob

import numpy as np

from ...functional import FlowAugmentor
from .base_dataset import BaseDataset


class Monkaa(BaseDataset):
    """
    Dataset Class for preparing the Monkaa Synthetic dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the Monkaa dataset
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
        super(Monkaa, self).__init__(
            augment, aug_params, is_prediction, init_seed, append_valid_mask
        )

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(**aug_params)

        image_list = []
        flow_list = []
        img_dir = os.path.join(root_dir, "frames_cleanpass")
        flow_dir = os.path.join(root_dir, "optical_flow")
        seqs_all = os.listdir(img_dir)
        for i, cur_seq in enumerate(seqs_all):
            for direction in ["into_future", "into_past"]:
                for cam in ["left", "right"]:
                    im_paths = sorted(
                        glob(os.path.join(img_dir, cur_seq, cam, "*.png"))
                    )
                    fl_paths = sorted(
                        glob(os.path.join(flow_dir, cur_seq, direction, cam, "*.pfm"))
                    )
                    if direction == "into_past":
                        im_paths = im_paths[::-1]
                        fl_paths = fl_paths[::-1]
                    for idx in range(len(im_paths) - 1):
                        flow_fn = fl_paths[idx]
                        if os.path.isfile(flow_fn):
                            image_list.append([im_paths[idx], im_paths[idx + 1]])
                            flow_list.append(fl_paths[idx])

        self.image_list = image_list
        self.flow_list = flow_list
