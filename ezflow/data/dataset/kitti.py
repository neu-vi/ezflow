import os.path as osp
from glob import glob

from ...config import configurable
from ...functional import SparseFlowAugmentor
from ..build import DATASET_REGISTRY
from .base_dataset import BaseDataset


@DATASET_REGISTRY.register()
class Kitti(BaseDataset):
    """
    Dataset Class for preparing the Kitti dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the HD1K dataset
    split : str, default : "training"
        specify the training or validation split
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
    norm_params : :obj:`dict`, optional
        The parameters for normalization
    flow_offset_params: :obj:`dict`, optional
        The parameters for adding bilinear interpolated weights surrounding each ground truth flow values.
    """

    @configurable
    def __init__(
        self,
        root_dir,
        split="training",
        is_prediction=False,
        init_seed=False,
        append_valid_mask=False,
        crop=False,
        crop_size=(256, 256),
        crop_type="center",
        augment=True,
        aug_params={
            "eraser_aug_params": {"enabled": False},
            "noise_aug_params": {"enabled": False},
            "flip_aug_params": {"enabled": False},
            "color_aug_params": {"enabled": False},
            "spatial_aug_params": {"enabled": False},
            "advanced_spatial_aug_params": {"enabled": False},
        },
        norm_params={"use": False},
        flow_offset_params={"use": False},
    ):
        super(Kitti, self).__init__(
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            sparse_transform=True,
            norm_params=norm_params,
            flow_offset_params=flow_offset_params,
        )
        assert (
            split.lower() == "training" or split.lower() == "validation"
        ), "Incorrect split values. Accepted split values: training, validation"

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = SparseFlowAugmentor(crop_size=crop_size, **aug_params)

        split = split.lower()
        if split == "validation":
            split = "testing"
            self.is_prediction = True

        root_dir = osp.join(root_dir, split)
        images1 = sorted(glob(osp.join(root_dir, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root_dir, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            self.image_list += [[img1, img2]]

        if not self.is_prediction:
            self.flow_list += sorted(glob(osp.join(root_dir, "flow_occ/*_10.png")))

    @classmethod
    def from_config(cls, cfg):
        return {
            "root_dir": cfg.ROOT_DIR,
            "split": cfg.SPLIT,
            "is_prediction": cfg.IS_PREDICTION,
            "init_seed": cfg.INIT_SEED,
            "append_valid_mask": cfg.APPEND_VALID_MASK,
            "crop": cfg.CROP.USE,
            "crop_size": cfg.CROP.SIZE,
            "crop_type": cfg.CROP.TYPE,
            "augment": cfg.AUGMENTATION.USE,
            "aug_params": cfg.AUGMENTATION.PARAMS,
            "norm_params": cfg.NORM_PARAMS,
            "flow_offset_params": cfg.FLOW_OFFSET_PARAMS,
        }
