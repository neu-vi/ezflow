import os
from glob import glob

from ...config import configurable
from ...functional import FlowAugmentor
from ..build import DATASET_REGISTRY
from .base_dataset import BaseDataset


@DATASET_REGISTRY.register()
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
        super(Monkaa, self).__init__(
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

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(crop_size=crop_size, **aug_params)

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

    @classmethod
    def from_config(cls, cfg):
        return {
            "root_dir": cfg.ROOT_DIR,
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
