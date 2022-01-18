import os.path as osp
from glob import glob

import numpy as np

from ...functional import FlowAugmentor
from .base_dataset import BaseDataset


class AutoFlow(BaseDataset):
    """
    Dataset Class for preparing the AutoFlow Synthetic dataset for training and validation.

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
        super(AutoFlow, self).__init__(
            augment, aug_params, is_prediction, init_seed, append_valid_mask
        )

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(**aug_params)

        scenes = [
            "static_40k_png_1_of_4",
            "static_40k_png_2_of_4",
            "static_40k_png_2_of_4",
            "static_40k_png_2_of_4",
        ]

        for scene in scenes:
            seqs = glob(osp.join(root_dir, scene, "*"))
            for s in seqs:
                images = sorted(glob(osp.join(s, "*.png")))
                flows = sorted(glob(osp.join(s, "*.flo")))
                if len(images) == 2:
                    assert len(flows) == 1
                    for i in range(len(flows)):
                        self.flow_list += [flows[i]]
                        self.image_list += [[images[i], images[i + 1]]]
