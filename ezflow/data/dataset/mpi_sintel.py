import os
import os.path as osp
from glob import glob

from ...config import configurable
from ...functional import FlowAugmentor
from ..build import DATASET_REGISTRY
from .base_dataset import BaseDataset


@DATASET_REGISTRY.register()
class MPISintel(BaseDataset):
    """
    Dataset Class for preparing the MPI Sintel Synthetic dataset for training and validation.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the MPI Sintel datasets
    split : str, default : "training"
        specify the training or validation split
    dstype : str, default : "clean"
        specify dataset type
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
        dstype="clean",
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
        super(MPISintel, self).__init__(
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            sparse_transform=False,
            norm_params=norm_params,
            flow_offset_params=flow_offset_params,
        )

        assert (
            split.lower() == "training" or split.lower() == "validation"
        ), "Incorrect split values. Accepted split values: training, validation"

        assert (
            dstype.lower() == "clean" or dstype.lower() == "final"
        ), "Incorrect dstype values. Accepted dstype values: clean, final"

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(crop_size=crop_size, **aug_params)

        split = split.lower()
        if split == "validation":
            split = "test"
            self.is_prediction = True

        image_root = osp.join(root_dir, split, dstype)
        flow_root = osp.join(root_dir, split, "flow")

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]

            if not self.is_prediction:
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

    @classmethod
    def from_config(cls, cfg):
        return {
            "root_dir": cfg.ROOT_DIR,
            "split": cfg.SPLIT,
            "dstype": cfg.DS_TYPE,
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


@DATASET_REGISTRY.register()
class MPISintelClean(MPISintel):
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
        super(MPISintelClean, self).__init__(
            root_dir=root_dir,
            split=split,
            dstype="clean",
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            norm_params=norm_params,
            flow_offset_params=flow_offset_params,
        )

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


@DATASET_REGISTRY.register()
class MPISintelFinal(MPISintel):
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
        super(MPISintelFinal, self).__init__(
            root_dir=root_dir,
            split=split,
            dstype="final",
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            norm_params=norm_params,
            flow_offset_params=flow_offset_params,
        )

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
