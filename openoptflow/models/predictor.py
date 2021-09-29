import torch
from torchvision import io

from .build import build_model


class DefaultPredictor:
    def __init__(
        self,
        model_name,
        model_cfg_path=None,
        model_cfg=None,
        model_weights_path=None,
        custom_cfg_file=False,
        default=False,
        data_transform=None,
        device="cpu",
    ):

        if model_cfg_path is not None:
            self.model = build_model(
                model_name,
                cfg_path=model_cfg_path,
                custom_cfg=custom_cfg_file,
                default=default,
                weights_path=model_weights_path,
            )

        elif default:
            self.model = build_model(
                model_name, default=True, weights_path=model_weights_path
            )

        else:
            assert (
                model_cfg is not None
            ), "Must provide either a path to a config file or a config object"
            self.model = build_model(
                model_name, cfg=model_cfg, weights_path=model_weights_path
            )

        self.model = self.model.eval()
        self.data_transform = data_transform
        self.device = torch.device(device)

    def __call__(self, img1, img2):

        if type(img1) == str:
            img1 = io.read_image(img1)
        if type(img2) == str:
            img2 = io.read_image(img2)

        if self.data_transform is not None:
            img1 = self.data_transform(img1)
            img2 = self.data_transform(img2)

        return self.model(img1, img2)
