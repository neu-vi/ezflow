from ..registry import FUNCTIONAL_REGISTRY
from .operations import *


@FUNCTIONAL_REGISTRY.register()
class FlowAugmentor:
    """
    Class for appyling a series of augmentations to a pair of images and a flow field

    Parameters
    ----------
    crop_type : str
        Type of crop to apply to the images; one of "center", "random".
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
        crop_type="center",
        color_aug_params={"aug_prob": 0.2},
        eraser_aug_params={"aug_prob": 0.5},
        spatial_aug_params={"aug_prob": 0.8},
    ):

        self.crop_size = crop_size
        self.crop_type = crop_type
        self.color_aug_params = color_aug_params
        self.eraser_aug_params = eraser_aug_params
        self.spatial_aug_params = spatial_aug_params

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
        img1, img2 = eraser_transform(img1, img2, **self.eraser_aug_params)
        img1, img2, flow = spatial_transform(
            img1, img2, flow, self.crop_size, self.crop_type, **self.spatial_aug_params
        )

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
            img1,
            img2,
            flow,
            valid,
            self.crop_size,
            self.crop_type,
            **self.spatial_aug_params
        )

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
