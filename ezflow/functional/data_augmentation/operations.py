from __future__ import division

import numbers
import pdb
import random

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import ColorJitter


def crop(
    img1,
    img2,
    flow,
    crop_size=(256, 256),
    crop_type="center",
    sparse_transform=False,
    valid=None,
):

    """
    Function to crop the images and flow field

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    valid : numpy.ndarray
        Valid flow mask
    crop_size : tuple
        Size of the crop
    crop_type : str
        Type of cropping
    sparse_transform : bool
        Whether to apply sparse transform

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    flow : numpy.ndarray
        Augmented flow field
    valid : numpy.ndarray
        Valid flow mask
    """

    if sparse_transform is True:
        assert valid is not None, "Valid flow mask is required for sparse transform"

    H, W = img1.shape[:2]

    y0 = 0
    x0 = 0
    if crop_type.lower() == "center":
        y0 = max(0, int(H / 2 - crop_size[0] / 2))
        x0 = max(0, int(W / 2 - crop_size[1] / 2))

    else:
        if sparse_transform is True:
            margin_y = 20
            margin_x = 50
            y0 = np.random.randint(0, img1.shape[0] - crop_size[0] + margin_y)
            x0 = np.random.randint(-margin_x, img1.shape[1] - crop_size[1] + margin_x)
            y0 = max(0, np.clip(y0, 0, img1.shape[0] - crop_size[0]))
            x0 = max(0, np.clip(x0, 0, img1.shape[1] - crop_size[1]))

        else:
            if img1.shape[0] - crop_size[0] > 0:
                y0 = max(0, np.random.randint(0, img1.shape[0] - crop_size[0]))

            if img1.shape[1] - crop_size[1] > 0:
                x0 = max(0, np.random.randint(0, img1.shape[1] - crop_size[1]))

    img1 = img1[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    img2 = img2[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    if flow is not None:
        flow = flow[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    if sparse_transform is True:
        valid = valid[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    return img1, img2, flow, valid


def color_transform(
    img1,
    img2,
    enabled=False,
    asymmetric_color_aug_prob=0.2,
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.5 / 3.14,
):
    """
    Photometric augmentation borrowed from RAFT https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    enabled : bool, default: False
        If True, applies color transform
    asymmetric_color_aug_prob : float
        Probability of applying asymetric color jitter augmentation
    brightness : float
        Brightness augmentation factor
    contrast : float
        Contrast augmentation factor
    saturation : float
        Saturation augmentation factor
    hue : float
        Hue augmentation factor

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    """
    if not enabled:
        return img1, img2

    aug = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    if np.random.rand() < asymmetric_color_aug_prob:
        img1 = np.array(aug(Image.fromarray(img1)), dtype=np.uint8)
        img2 = np.array(aug(Image.fromarray(img2)), dtype=np.uint8)

    else:
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

    return img1, img2


def eraser_transform(img1, img2, enabled=False, bounds=[50, 100], aug_prob=0.5):
    """
    Occlusion augmentation borrowed from RAFT https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    enabled : bool, default: False
        If True, applies eraser transform
    bounds : :obj:`list` of :obj:`int`
        Bounds of the eraser
    aug_prob : float
        Probability of applying the augmentation

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    """
    if not enabled:
        return img1, img2

    H, W = img1.shape[:2]

    if np.random.rand() < aug_prob:

        mean_color = np.mean(img2.reshape(-1, 3), axis=0)

        for _ in range(np.random.randint(1, 3)):
            x0 = np.random.randint(0, W)
            y0 = np.random.randint(0, H)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])
            img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

    return img1, img2


def spatial_transform(
    img1,
    img2,
    flow,
    crop_size,
    enabled=False,
    aug_prob=0.8,
    stretch_prob=0.8,
    max_stretch=0.2,
    min_scale=-0.2,
    max_scale=0.5,
):
    """
    Simple set of spatial augmentation borrowed from RAFT https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py

    Includes random scaling and stretch.

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    crop_size : :obj:`list` of :obj:`int`
        Size of the crop
    enabled : bool, default: False
        If True, applies spatial transform
    aug_prob : float
        Probability of applying the augmentation
    stretch_prob : float
        Probability of applying the stretch transform
    max_stretch : float
        Maximum stretch factor
    min_scale : float
        Minimum scale factor
    max_scale : float
        Maximum scale factor

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    flow : numpy.ndarray
        Augmented flow field
    """
    if not enabled:
        return img1, img2, flow

    H, W = img1.shape[:2]

    scale = 2 ** np.random.uniform(min_scale, max_scale)
    scale_x = scale
    scale_y = scale

    if np.random.rand() < stretch_prob:
        scale_x *= 2 ** np.random.uniform(-max_stretch, max_stretch)
        scale_y *= 2 ** np.random.uniform(-max_stretch, max_stretch)

    min_scale = np.maximum((crop_size[0] + 8) / float(H), (crop_size[1] + 8) / float(W))

    scale_x = np.clip(scale_x, min_scale, None)
    scale_y = np.clip(scale_y, min_scale, None)

    if np.random.rand() < aug_prob:

        img1 = cv2.resize(
            img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
        )
        img2 = cv2.resize(
            img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
        )
        flow = cv2.resize(
            flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
        )
        flow = flow * [scale_x, scale_y]

    return img1, img2, flow


def flip_transform(
    img1, img2, flow, valid=None, enabled=False, h_flip_prob=0.5, v_flip_prob=0.1
):
    """
    Flip augmentation borrowed from RAFT https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    valid : numpy.ndarray, default: None
            Valid Flow field
    enabled : bool, default: False
        If True, applies flip transform
    h_flip_prob : float, default=0.5
        Probability of applying the horizontal flip transform
    v_flip_prob : float, default=0.1
        Probability of applying the vertical flip transform

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Flipped image 1
    img2 : PIL Image or numpy.ndarray
        Flipped image 2
    flow : numpy.ndarray
        Flipped flow field
    valid : numpy.ndarray, default: None
            Valid Flow field
    """

    if not enabled:
        return img1, img2, flow, valid

    if np.random.rand() < h_flip_prob:
        img1 = img1[:, ::-1]
        img2 = img2[:, ::-1]
        flow = flow[:, ::-1] * [-1.0, 1.0]
        valid = valid[:, ::-1] if valid is not None else None

    if np.random.rand() < v_flip_prob:
        img1 = img1[::-1, :]
        img2 = img2[::-1, :]
        flow = flow[::-1, :] * [1.0, -1.0]
        valid = valid[::-1, :] if valid is not None else None

    return img1, img2, flow, valid


def resize_sparse_flow_map(flow, valid, fx=1.0, fy=1.0):
    """
    Resize flow field and valid flow by the scaling factor of fx and fy

    Parameters
    -----------
    flow : numpy.ndarray
            Flow field
    valid : numpy.ndarray
            Valid Flow field
    fx : float
        Scaling factor along x
    fy : float
        Scaling factor along y

    Returns
    -------
    flow : numpy.ndarray
            Flow field
    valid : numpy.ndarray
            Valid Flow field
    """
    H, W = flow.shape[:2]
    coords = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack(coords, axis=-1)

    coords = coords.reshape(-1, 2).astype(np.float32)
    flow = flow.reshape(-1, 2).astype(np.float32)
    valid = valid.reshape(-1).astype(np.float32)

    coords0 = coords[valid >= 1]
    flow0 = flow[valid >= 1]

    H1 = int(round(H * fy))
    W1 = int(round(W * fx))

    coords1 = coords0 * [fx, fy]
    flow1 = flow0 * [fx, fy]

    xx = np.round(coords1[:, 0]).astype(np.int32)
    yy = np.round(coords1[:, 1]).astype(np.int32)

    v = (xx > 0) & (xx < W1) & (yy > 0) & (yy < H1)
    xx = xx[v]
    yy = yy[v]
    flow1 = flow1[v]

    flow_img = np.zeros([H1, W1, 2], dtype=np.float32)
    valid_img = np.zeros([H1, W1], dtype=np.int32)

    flow_img[yy, xx] = flow1
    valid_img[yy, xx] = 1

    return flow_img, valid_img


def sparse_spatial_transform(
    img1,
    img2,
    flow,
    valid,
    crop_size,
    enabled=False,
    aug_prob=0.8,
    min_scale=-0.2,
    max_scale=0.5,
    flip=True,
    h_flip_prob=0.5,
):
    """
    Sparse spatial augmentation.

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    valid : numpy.ndarray
        Valid flow field
    crop_size : :obj:`list` of :obj:`int`
        Size of the crop
    aug_prob : float
        Probability of applying the augmentation
    min_scale : float
        Minimum scale factor
    max_scale : float
        Maximum scale factor
    flip : bool
        Whether to apply the flip transform
    h_flip_prob : float
        Probability of applying the horizontal flip transform
    v_flip_prob : float
        Probability of applying the vertical flip transform

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    flow : numpy.ndarray
        Augmented flow field
    valid : numpy.ndarray
        Valid flow field
    """
    if not enabled:
        return img1, img2, flow, valid

    H, W = img1.shape[:2]
    min_scale = np.maximum((crop_size[0] + 1) / float(H), (crop_size[1] + 1) / float(W))

    scale = 2 ** np.random.uniform(min_scale, max_scale)
    scale_x = np.clip(scale, min_scale, None)
    scale_y = np.clip(scale, min_scale, None)

    if np.random.rand() < aug_prob:

        img1 = cv2.resize(
            img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
        )
        img2 = cv2.resize(
            img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
        )
        flow, valid = resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

    return img1, img2, flow, valid


class Normalize:
    """
    A class to return Normalized Image.

    Parameters
    -----------
    use : boolean
        Whether to normalize image or not
    mean : list
        The list of mean values to be substracted from each image channel
    std : list
        The list of std values with which to divide each image channel by
    """

    def __init__(self, use=False, mean=[0, 0, 0], std=[255.0, 255.0, 255.0]):
        self.use = use
        self.mean = mean
        self.std = std
        self.normalize = transforms.Compose(
            [
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, img1, img2):
        if self.use:
            return self.normalize(img1), self.normalize(img2)
        return img1, img2


def noise_transform(img1, img2, enabled=False, aug_prob=0.5, noise_std_range=0.06):
    """
    Applies random noise augmentation from a gaussian distribution borrowed from VCN:
    https://github.com/gengshan-y/VCN/blob/master/dataloader/flow_transforms.py

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    enabled : bool, default: False
        If True, applies noise transform
    aug_prob : float
        Probability of applying the augmentation
    noise_std_range : float
        Standard deviation of the noise

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    """

    if not enabled:
        return img1, img2

    if np.random.rand() < aug_prob:
        noise = np.random.uniform(0, noise_std_range * 255.0)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        img1 += np.random.normal(0, noise, img1.shape)
        img2 += np.random.normal(0, noise, img2.shape)

        img1 = np.clip(img1, 0.0, 255.0)
        img2 = np.clip(img2, 0.0, 255.0)

    return img1, img2


class AdvancedSpatialTransform(object):
    """
    Advanced set of spatial transformations borrowed from:

    1. VCN: https://github.com/gengshan-y/VCN/blob/master/dataloader/flow_transforms.py
    2. Autoflow: https://github.com/google-research/opticalflow-autoflow/blob/main/src/dataset_lib/augmentations/spatial_aug.py

    This set of augmentations include random scaling, stretch, rotation, translation and out-of-boundary cropping.

    Parameters
    -----------
    crop_size : :obj:`list` of :obj:`int`
        Size of the crop
    enabled : bool, default: False
        If True, applies flip transform
    scale1 : float, default : 0.3
        Scale factor 1
    scale1 : float, default : 0.1
        Scale factor 2
    rotate : float, default : 0.4
        Rotate factor
    translate : float, default : 0.4
        Translate factor
    stretch : float, default : 0.3
        Stretch factor
    h_flip_prob : float, default=0.5
        Probability of applying the horizontal flip transform

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Flipped image 1
    img2 : PIL Image or numpy.ndarray
        Flipped image 2
    flow : numpy.ndarray
        Flipped flow field
    """

    def __init__(
        self,
        crop,
        enabled=False,
        scale1=0.3,
        scale2=0.1,
        rotate=0.4,
        translate=0.4,
        stretch=0.3,
        h_flip_prob=0.5,
        schedule_coeff=1,
        order=1,
        enable_out_of_boundary_crop=False,
    ):
        self.enabled = enabled
        self.crop = crop
        self.scale = [scale1, 0.03, scale2]
        self.rot = [rotate, 0.03] if rotate != 0 else None
        self.trans = [translate, 0.03] if translate != 0 else None
        self.squeeze = [stretch, 0.0] if stretch != 0 else None
        self.h_flip_prob = h_flip_prob
        self.t = np.zeros(6)
        self.schedule_coeff = schedule_coeff
        self.order = order
        self.black = enable_out_of_boundary_crop

    def to_identity(self):
        self.t[0] = 1
        self.t[2] = 0
        self.t[4] = 0
        self.t[1] = 0
        self.t[3] = 1
        self.t[5] = 0

    def left_multiply(self, u0, u1, u2, u3, u4, u5):
        result = np.zeros(6)
        result[0] = self.t[0] * u0 + self.t[1] * u2
        result[1] = self.t[0] * u1 + self.t[1] * u3

        result[2] = self.t[2] * u0 + self.t[3] * u2
        result[3] = self.t[2] * u1 + self.t[3] * u3

        result[4] = self.t[4] * u0 + self.t[5] * u2 + u4
        result[5] = self.t[4] * u1 + self.t[5] * u3 + u5
        self.t = result

    def inverse(self):
        result = np.zeros(6)
        a = self.t[0]
        c = self.t[2]
        e = self.t[4]
        b = self.t[1]
        d = self.t[3]
        f = self.t[5]

        denom = a * d - b * c

        result[0] = d / denom
        result[1] = -b / denom
        result[2] = -c / denom
        result[3] = a / denom
        result[4] = (c * f - d * e) / denom
        result[5] = (b * e - a * f) / denom

        return result

    def grid_transform(self, meshgrid, t, normalize=True, gridsize=None):
        if gridsize is None:
            h, w = meshgrid[0].shape
        else:
            h, w = gridsize
        vgrid = torch.cat(
            [
                (meshgrid[0] * t[0] + meshgrid[1] * t[2] + t[4])[:, :, np.newaxis],
                (meshgrid[0] * t[1] + meshgrid[1] * t[3] + t[5])[:, :, np.newaxis],
            ],
            -1,
        )
        if normalize:
            vgrid[:, :, 0] = 2.0 * vgrid[:, :, 0] / max(w - 1, 1) - 1.0
            vgrid[:, :, 1] = 2.0 * vgrid[:, :, 1] / max(h - 1, 1) - 1.0
        return vgrid

    def __call__(self, img1, img2, target):
        """
        Parameters
        -----------
        img1 : PIL Image or numpy.ndarray
            First of the pair of images
        img2 : PIL Image or numpy.ndarray
            Second of the pair of images
        target : numpy.ndarray
            Flow field

        Returns
        -------
        img1 : PIL Image or numpy.ndarray
            Flipped image 1
        img2 : PIL Image or numpy.ndarray
            Flipped image 2
        flow : numpy.ndarray
            Flipped flow field
        """
        if not self.enabled:
            return img1, img2, target

        inputs = [img1, img2]
        h, w, _ = inputs[0].shape
        th, tw = self.crop
        meshgrid = torch.meshgrid(
            [torch.Tensor(range(th)), torch.Tensor(range(tw))], indexing="ij"
        )[::-1]
        cornergrid = torch.meshgrid(
            [torch.Tensor([0, th - 1]), torch.Tensor([0, tw - 1])], indexing="ij"
        )[::-1]

        for i in range(50):
            # im0
            self.to_identity()

            if np.random.binomial(1, self.h_flip_prob):
                mirror = True
            else:
                mirror = False

            if mirror:
                self.left_multiply(-1, 0, 0, 1, 0.5 * tw, -0.5 * th)
            else:
                self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            scale0 = 1
            scale1 = 1
            squeeze0 = 1
            squeeze1 = 1
            if not self.rot is None:
                rot0 = np.random.uniform(-self.rot[0], +self.rot[0])
                rot1 = (
                    np.random.uniform(
                        -self.rot[1] * self.schedule_coeff,
                        self.rot[1] * self.schedule_coeff,
                    )
                    + rot0
                )
                self.left_multiply(
                    np.cos(rot0), np.sin(rot0), -np.sin(rot0), np.cos(rot0), 0, 0
                )
            if not self.trans is None:
                trans0 = np.random.uniform(-self.trans[0], +self.trans[0], 2)
                trans1 = (
                    np.random.uniform(
                        -self.trans[1] * self.schedule_coeff,
                        +self.trans[1] * self.schedule_coeff,
                        2,
                    )
                    + trans0
                )
                self.left_multiply(1, 0, 0, 1, trans0[0] * tw, trans0[1] * th)
            if not self.squeeze is None:
                squeeze0 = np.exp(np.random.uniform(-self.squeeze[0], self.squeeze[0]))
                squeeze1 = (
                    np.exp(
                        np.random.uniform(
                            -self.squeeze[1] * self.schedule_coeff,
                            self.squeeze[1] * self.schedule_coeff,
                        )
                    )
                    * squeeze0
                )
            if not self.scale is None:
                scale0 = np.exp(
                    np.random.uniform(
                        self.scale[2] - self.scale[0], self.scale[2] + self.scale[0]
                    )
                )
                scale1 = (
                    np.exp(
                        np.random.uniform(
                            -self.scale[1] * self.schedule_coeff,
                            self.scale[1] * self.schedule_coeff,
                        )
                    )
                    * scale0
                )
            self.left_multiply(
                1.0 / (scale0 * squeeze0), 0, 0, 1.0 / (scale0 / squeeze0), 0, 0
            )

            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat0 = self.t.copy()

            # im1
            self.to_identity()
            if mirror:
                self.left_multiply(-1, 0, 0, 1, 0.5 * tw, -0.5 * th)
            else:
                self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            if not self.rot is None:
                self.left_multiply(
                    np.cos(rot1), np.sin(rot1), -np.sin(rot1), np.cos(rot1), 0, 0
                )
            if not self.trans is None:
                self.left_multiply(1, 0, 0, 1, trans1[0] * tw, trans1[1] * th)
            self.left_multiply(
                1.0 / (scale1 * squeeze1), 0, 0, 1.0 / (scale1 / squeeze1), 0, 0
            )
            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat1 = self.t.copy()
            transmat1_inv = self.inverse()

            if self.black:
                # black augmentation, allowing 0 values in the input images
                # https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/black_augmentation_layer.cu
                break
            else:
                if (
                    (
                        self.grid_transform(
                            cornergrid, transmat0, gridsize=[float(h), float(w)]
                        ).abs()
                        > 1
                    ).sum()
                    + (
                        self.grid_transform(
                            cornergrid, transmat1, gridsize=[float(h), float(w)]
                        ).abs()
                        > 1
                    ).sum()
                ) == 0:
                    break
        if i == 49:
            # print("max_iter in augmentation")
            self.to_identity()
            self.left_multiply(1, 0, 0, 1, -0.5 * tw, -0.5 * th)
            self.left_multiply(1, 0, 0, 1, 0.5 * w, 0.5 * h)
            transmat0 = self.t.copy()
            transmat1 = self.t.copy()

        # do the real work
        vgrid = self.grid_transform(meshgrid, transmat0, gridsize=[float(h), float(w)])
        inputs_0 = F.grid_sample(
            torch.Tensor(inputs[0]).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
        )[0].permute(1, 2, 0)
        if self.order == 0:
            target_0 = F.grid_sample(
                torch.Tensor(target).permute(2, 0, 1)[np.newaxis],
                vgrid[np.newaxis],
                mode="nearest",
            )[0].permute(1, 2, 0)
        else:
            target_0 = F.grid_sample(
                torch.Tensor(target).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
            )[0].permute(1, 2, 0)

        mask_0 = target[:, :, 2:3].copy()
        mask_0[mask_0 == 0] = np.nan
        if self.order == 0:
            mask_0 = F.grid_sample(
                torch.Tensor(mask_0).permute(2, 0, 1)[np.newaxis],
                vgrid[np.newaxis],
                mode="nearest",
            )[0].permute(1, 2, 0)
        else:
            mask_0 = F.grid_sample(
                torch.Tensor(mask_0).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
            )[0].permute(1, 2, 0)
        mask_0[torch.isnan(mask_0)] = 0

        vgrid = self.grid_transform(meshgrid, transmat1, gridsize=[float(h), float(w)])
        inputs_1 = F.grid_sample(
            torch.Tensor(inputs[1]).permute(2, 0, 1)[np.newaxis], vgrid[np.newaxis]
        )[0].permute(1, 2, 0)

        # flow
        pos = target_0[:, :, :2] + self.grid_transform(
            meshgrid, transmat0, normalize=False
        )
        pos = self.grid_transform(pos.permute(2, 0, 1), transmat1_inv, normalize=False)
        if target_0.shape[2] >= 4:
            # scale
            exp = target_0[:, :, 3:] * scale1 / scale0
            target = torch.cat(
                [
                    (pos[:, :, 0] - meshgrid[0]).unsqueeze(-1),
                    (pos[:, :, 1] - meshgrid[1]).unsqueeze(-1),
                    mask_0,
                    exp,
                ],
                -1,
            )
        else:
            target = torch.cat(
                [
                    (pos[:, :, 0] - meshgrid[0]).unsqueeze(-1),
                    (pos[:, :, 1] - meshgrid[1]).unsqueeze(-1),
                    mask_0,
                ],
                -1,
            )
        #                               target_0[:,:,2].unsqueeze(-1) ], -1)
        inputs = [np.asarray(inputs_0), np.asarray(inputs_1)]
        target = np.asarray(target)

        return inputs[0], inputs[1], target
