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
        color_aug_params={"aug_prob": 0.2},
        eraser_aug_params={"aug_prob": 0.5},
        spatial_aug_params={"aug_prob": 0.8},
        translate_params={"aug_prob": 0.8},
        rotate_params={"aug_prob": 0.8},
        noise_params={"aug_prob": 0.5},
        spatial_params={},
        chromatic_params={},
    ):

        self.crop_size = crop_size
        self.color_aug_params = color_aug_params
        self.eraser_aug_params = eraser_aug_params
        self.spatial_aug_params = spatial_aug_params
        self.translate_params = translate_params
        self.rotate_params = rotate_params
        self.noise_params = noise_params

        self.spatial_params = spatial_params

        self.chromatic_params = {
            "enabled": chromatic_params["enabled"],
            "lmult_pow": [0.4 * chromatic_params["lmult_factor"], 0, -0.2],
            "lmult_mult": [0.4 * chromatic_params["lmult_factor"], 0, 0],
            "lmult_add": [0.03 * chromatic_params["lmult_factor"], 0, 0],
            "sat_pow": [0.4 * chromatic_params["sat_factor"], 0, 0],
            "sat_mult": [0.5 * chromatic_params["sat_factor"], 0, -0.3],
            "sat_add": [0.03 * chromatic_params["sat_factor"], 0, 0],
            "col_pow": [0.4 * chromatic_params["col_factor"], 0, 0],
            "col_mult": [0.2 * chromatic_params["col_factor"], 0, 0],
            "col_add": [0.02 * chromatic_params["col_factor"], 0, 0],
            "ladd_pow": [0.4 * chromatic_params["ladd_factor"], 0, 0],
            "ladd_mult": [0.4 * chromatic_params["ladd_factor"], 0, 0],
            "ladd_add": [0.04 * chromatic_params["ladd_factor"], 0, 0],
            "col_rotate": [1.0 * chromatic_params["col_rot_factor"], 0, 0],
            "gamma": 0.02,
            "brightness": 0.02,
            "contrast": 0.02,
            "color": 0.02,
            "schedule_coeff": 1,
        }

        self.spatial_transform = SpatialAug(crop=self.crop_size, **self.spatial_params)
        self.chromatic_transform = PCAAug(**self.chromatic_params)

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
        img1, img2 = self.chromatic_transform(img1, img2)

        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1, img2, flow = translate_transform(
            img1, img2, flow, **self.translate_params
        )
        img1, img2, flow = rotate_transform(img1, img2, flow, **self.rotate_params)
        img1, img2, flow = spatial_transform(
            img1, img2, flow, self.crop_size, **self.spatial_aug_params
        )

        img1, img2 = noise_transform(img1, img2, **self.noise_params)
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
