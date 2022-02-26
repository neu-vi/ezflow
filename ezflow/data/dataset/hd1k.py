import os.path as osp
from glob import glob

from ...functional import SparseFlowAugmentor
from .base_dataset import BaseDataset


class HD1K(BaseDataset):
    """
    Dataset Class for preparing the HD1K dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the HD1K dataset
    is_prediction : bool, default : False
        If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    init_seed : bool, default : False
        If True, sets random seed to worker
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    crop: bool, default : True
        Whether to perform cropping
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    crop_type : :obj:`str`, default : 'center'
        The type of croppping to be performed, one of "center", "random"
    augment : bool, default : True
        If True, applies data augmentation
    aug_params : :obj:`dict`, optional
        The parameters for data augmentation
    """

    def __init__(
        self,
        root_dir,
        is_prediction=False,
        init_seed=False,
        append_valid_mask=False,
        crop=False,
        crop_size=(256, 256),
        crop_type="center",
        augment=True,
        aug_params={
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
        },
    ):
        super(HD1K, self).__init__(
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            sparse_transform=True,
        )

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = SparseFlowAugmentor(crop_size=crop_size, **aug_params)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(osp.join(root_dir, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(osp.join(root_dir, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1
