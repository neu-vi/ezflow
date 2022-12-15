from ..registry import FUNCTIONAL_REGISTRY
from .operations import *


@FUNCTIONAL_REGISTRY.register()
class FlowAugmentor:
    """
    Class for appyling a series of augmentations to a pair of images and a flow field

    Parameters
    ----------
    crop_size : int
        Size of the crop to be applied to the images.
    color_aug_params : dict
        Parameters for the color augmentation.
    eraser_aug_params : dict
        Parameters for the eraser augmentation.
    spatial_aug_params : dict
        Parameters for the spatial augmentation.
    """

    def __init__(
        self,
        crop_size,
        eraser_aug_params={"enabled": False, "aug_prob": 0.5, "bounds": [50, 100]},
        noise_aug_params={"enabled": False, "aug_prob": 0.5, "noise_std_range": 0.06},
        flip_aug_params={"enabled": False, "h_flip_prob": 0.5, "v_flip_prob": 0.1},
        color_aug_params={
            "enabled": False,
            "asymmetric_color_aug_prob": 0.2,
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.15915494309189535,
        },
        spatial_aug_params={
            "enabled": False,
            "aug_prob": 0.8,
            "stretch_prob": 0.8,
            "min_scale": -0.1,
            "max_scale": 1.0,
            "max_stretch": 0.2,
        },
        advanced_spatial_aug_params={
            "enabled": False,
            "scale1": 0.3,
            "scale2": 0.1,
            "rotate": 0.4,
            "translate": 0.4,
            "stretch": 0.3,
            "enable_out_of_boundary_crop": False,
        },
    ):

        self.crop_size = crop_size
        self.color_aug_params = color_aug_params
        self.eraser_aug_params = eraser_aug_params
        self.spatial_aug_params = spatial_aug_params
        self.noise_aug_params = noise_aug_params
        self.flip_aug_params = flip_aug_params

        self.advanced_spatial_aug_params = advanced_spatial_aug_params
        self.advanced_spatial_aug_params["h_flip_prob"] = (
            flip_aug_params["h_flip_prob"] if "h_flip_prob" in flip_aug_params else 0.0
        )
        self.advanced_spatial_transform = AdvancedSpatialTransform(
            crop=self.crop_size, **self.advanced_spatial_aug_params
        )

        if self.advanced_spatial_aug_params["enabled"]:
            # Disable spatial transform and horizontal flip if advanced spatial transforms are used
            self.spatial_aug_params["enabled"] = False
            self.flip_aug_params["h_flip_prob"] = 0.0

    def __call__(self, img1, img2, flow, valid=None):
        """
        Applies the augmentations to the pair of images and the flow field.

        Parameters
        ----------
        img1 : numpy.ndarray
            First image
        img2 : numpy.ndarray
            Second image
        flow : numpy.ndarray
            Flow field
        valid : default: None
            None object

        Returns
        -------
        img1 : numpy.ndarray
            First image
        img2 : numpy.ndarray
            Second image
        flow : numpy.ndarray
            Flow field
        valid : None
            None object
        """

        img1, img2 = color_transform(img1, img2, **self.color_aug_params)

        img1, img2, flow = self.advanced_spatial_transform(img1, img2, flow)

        img1, img2, flow = spatial_transform(
            img1, img2, flow, self.crop_size, **self.spatial_aug_params
        )

        img1, img2, flow = flip_transform(img1, img2, flow, **self.flip_aug_params)

        img1, img2 = noise_transform(img1, img2, **self.noise_aug_params)
        img1, img2 = eraser_transform(img1, img2, **self.eraser_aug_params)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow, None


@FUNCTIONAL_REGISTRY.register()
class SparseFlowAugmentor(FlowAugmentor):
    """
    Class for appyling a series of augmentations to a pair of images, flow field and valid flow field.

    Parameters
    ----------
    crop_size : int
        Size of the crop to be applied to the images.
    color_aug_params : dict
        Parameters for the color augmentation.
    eraser_aug_params : dict
        Parameters for the eraser augmentation.
    spatial_aug_params : dict
        Parameters for the spatial augmentation.
    """

    def __call__(self, img1, img2, flow, valid):
        """
        Applies the augmentations to the pair of images and the flow field.

        Parameters
        ----------
        img1 : numpy.ndarray
            First image
        img2 : numpy.ndarray
            Second image
        flow : numpy.ndarray
            Flow field
        valid : numpy.ndarray
            Valid Flow field

        Returns
        -------
        img1 : numpy.ndarray
            First image
        img2 : numpy.ndarray
            Second image
        flow : numpy.ndarray
            Flow field
        valid : numpy.ndarray
            Valid Flow field
        """
        img1, img2 = color_transform(img1, img2, **self.color_aug_params)
        img1, img2 = eraser_transform(img1, img2, **self.eraser_aug_params)
        img1, img2, flow, valid = sparse_spatial_transform(
            img1, img2, flow, valid, self.crop_size, **self.spatial_aug_params
        )

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
