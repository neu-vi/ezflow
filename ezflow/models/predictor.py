import torch
from torchvision import io

from ..utils import InputPadder
from .build import build_model


class Predictor:
    """
    A class that uses an instance of an optical flow estimation model to predict flow between two images

    Parameters
    ----------
    model_name : str
        The name of the optical flow estimation model to use
    model_cfg_path : str, optional
        The path to the config file for the optical flow estimation model, by default None in which case the default config is used
    model_cfg : CfgNode object, optional
        The config object for the optical flow estimation model, by default None
    model_weights_path : str, optional
        The path to the weights file for the optical flow estimation model
    custom_cfg_file : bool, optional
        Whether the config file is a custom config file or one one of the configs included in EzFlow, by default False
    default : bool, optional
        Whether to use the default config for the model
    data_transform : torchvision.transforms object, optional
        The data transform to apply to the images before passing them to the model, by default None
    device : str, optional
        The device to use for the model, by default "cpu"
    flow_scale : float, optional
        The scale to apply to the predicted flow, by default 1.0
    pad_divisor : int, optional
        The divisor to make the image dimensions evenly divisible by using padding, by default 1
    """

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
        flow_scale=1.0,
        pad_divisor=1,
    ):

        self.flow_scale = flow_scale
        self.pad_divisor = pad_divisor

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
        """
        Runs the prediction on the two images

        Parameters
        ----------
        img1 : torch.Tensor or str
            The first image to predict flow from
        img2 : torch.Tensor or str
            The second image to predict flow to

        Returns
        -------
        torch.Tensor
            The predicted flow
        """

        if type(img1) == str:
            img1 = io.read_image(img1)
        if type(img2) == str:
            img2 = io.read_image(img2)

        if self.data_transform is not None:
            img1 = self.data_transform(img1)
            img2 = self.data_transform(img2)

        padder = InputPadder(img1.shape, divisor=self.pad_divisor)
        img1, img2 = padder.pad(img1, img2)

        flow_pred = self.model(img1, img2)
        flow_pred = padder.unpad(flow_pred)
        flow_pred = flow_pred * self.flow_scale

        return flow_pred
