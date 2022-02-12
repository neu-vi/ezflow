import os
from glob import glob

from ...functional import FlowAugmentor
from .base_dataset import BaseDataset


class Driving(BaseDataset):
    """
    Dataset Class for preparing the Driving dataset for training and validation.

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
        super(Driving, self).__init__(
            init_seed=init_seed,
            is_prediction=is_prediction,
            append_valid_mask=append_valid_mask,
            crop=crop,
            crop_size=crop_size,
            crop_type=crop_type,
            augment=augment,
            aug_params=aug_params,
            sparse_transform=False,
        )

        self.is_prediction = is_prediction
        self.append_valid_mask = append_valid_mask

        if augment:
            self.augmentor = FlowAugmentor(crop_size=crop_size, **aug_params)

        image_list = []
        flow_list = []
        img_dir = os.path.join(root_dir, "frames_cleanpass")
        flow_dir = os.path.join(root_dir, "optical_flow")

        for fcl in os.listdir(img_dir):
            seqs_all = os.listdir(os.path.join(img_dir, fcl))

            for _, cur_seq in enumerate(seqs_all):
                for fs in os.listdir(os.path.join(img_dir, fcl, cur_seq)):
                    for direction in ["into_future", "into_past"]:
                        for cam in ["left", "right"]:

                            im_paths = sorted(
                                glob(
                                    os.path.join(
                                        img_dir, fcl, cur_seq, fs, cam, "*.png"
                                    )
                                )
                            )
                            fl_paths = sorted(
                                glob(
                                    os.path.join(
                                        flow_dir,
                                        fcl,
                                        cur_seq,
                                        fs,
                                        direction,
                                        cam,
                                        "*.pfm",
                                    )
                                )
                            )
                            if direction == "into_past":
                                im_paths = im_paths[::-1]
                                fl_paths = fl_paths[::-1]
                            for idx in range(len(im_paths) - 1):
                                flow_fn = fl_paths[idx]
                                if os.path.isfile(flow_fn):
                                    image_list.append(
                                        [im_paths[idx], im_paths[idx + 1]]
                                    )
                                    flow_list.append(fl_paths[idx])

        self.image_list = image_list
        self.flow_list = flow_list
