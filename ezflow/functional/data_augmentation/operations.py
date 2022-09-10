import cv2
import numpy as np
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
from PIL import Image
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
    flow = flow[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    if sparse_transform is True:
        valid = valid[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    return img1, img2, flow, valid


def color_transform(
    img1,
    img2,
    aug_prob=0.2,
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.5 / 3.14,
):
    """
    Photometric augmentation

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    aug_prob : float
        Probability of applying assymetric color jitter augmentation
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

    aug = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    if np.random.rand() < aug_prob:
        img1 = np.array(aug(Image.fromarray(img1)), dtype=np.uint8)
        img2 = np.array(aug(Image.fromarray(img2)), dtype=np.uint8)

    else:
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

    return img1, img2


def eraser_transform(img1, img2, bounds=[50, 100], aug_prob=0.5):
    """
    Occlusion augmentation
    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
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
    aug_prob=0.8,
    stretch_prob=0.8,
    max_stretch=0.2,
    min_scale=-0.2,
    max_scale=0.5,
    flip=True,
    h_flip_prob=0.5,
    v_flip_prob=0.1,
):
    """
    Spatial augmentation

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
    """

    H, W = img1.shape[:2]
    min_scale = np.maximum((crop_size[0] + 8) / float(H), (crop_size[1] + 8) / float(W))

    scale = 2 ** np.random.uniform(min_scale, max_scale)
    scale_x = scale
    scale_y = scale

    if np.random.rand() < stretch_prob:
        scale_x *= 2 ** np.random.uniform(-max_stretch, max_stretch)
        scale_y *= 2 ** np.random.uniform(-max_stretch, max_stretch)

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

    if flip:
        if np.random.rand() < h_flip_prob:
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]

        if np.random.rand() < v_flip_prob:
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]

    return img1, img2, flow


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

    if flip:
        if np.random.rand() < h_flip_prob:
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
            valid = valid[:, ::-1]

    return img1, img2, flow, valid


def translate_transform(
    img1,
    img2,
    flow,
    aug_prob=0.8,
    translate=10,
):
    """
    Translation augmentation.

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    aug_prob : float
        Probability of applying the augmentation
    translate : int
        Pixels by which image will be translated

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    flow : numpy.ndarray
        Augmented flow field
    """
    H, W = img1.shape[:2]

    max_t_x = translate
    max_t_y = translate

    t_x = np.random.randint(-1 * max_t_x, max_t_x)
    t_y = np.random.randint(-1 * max_t_y, max_t_y)

    if t_x == 0 and t_y == 0:
        return img1, img2, flow

    if np.random.rand() < aug_prob:

        x1, x2, x3, x4 = max(0, t_x), min(W + t_x, W), max(0, -t_x), min(W - t_x, W)
        y1, y2, y3, y4 = max(0, t_y), min(H + t_y, H), max(0, -t_y), min(H - t_y, H)

        img1 = img1[y1:y2, x1:x2]
        img2 = img2[y3:y4, x3:x4]
        flow = flow[y1:y2, x1:x2]
        flow[:, :, 0] += t_x
        flow[:, :, 1] += t_y

    return img1, img2, flow


def rotate_transform(
    img1,
    img2,
    flow,
    aug_prob=0.8,
    degrees=10,
    delta=0,
):
    """
    Rotation augmentation.
    (Referenced from Clement Picard)

    Parameters
    -----------
    img1 : PIL Image or numpy.ndarray
        First of the pair of images
    img2 : PIL Image or numpy.ndarray
        Second of the pair of images
    flow : numpy.ndarray
        Flow field
    aug_prob : float
        Probability of applying the augmentation
    degrees : int
        Angle by which image is to rotated
    delta: int
        Assigns angle range of degrees-delta to degrees+delta

    Returns
    -------
    img1 : PIL Image or numpy.ndarray
        Augmented image 1
    img2 : PIL Image or numpy.ndarray
        Augmented image 2
    flow : numpy.ndarray
        Augmented flow field
    """

    angle = np.random.uniform(-degrees, degrees)
    diff = np.random.uniform(-delta, delta)
    angle1 = angle - diff / 2
    angle2 = angle + diff / 2
    angle1_rad = angle1 * np.pi / 180
    diff_rad = diff * np.pi / 180

    H, W = img1.shape[:2]

    warped_coords = np.mgrid[:W, :H].T + flow
    warped_coords -= np.array([W / 2, H / 2])

    warped_coords_rot = np.zeros_like(flow)

    warped_coords_rot[..., 0] = (np.cos(diff_rad) - 1) * warped_coords[..., 0] + np.sin(
        diff_rad
    ) * warped_coords[..., 1]

    warped_coords_rot[..., 1] = (
        -np.sin(diff_rad) * warped_coords[..., 0]
        + (np.cos(diff_rad) - 1) * warped_coords[..., 1]
    )

    if np.random.rand() < aug_prob:

        flow += warped_coords_rot

        img1 = ndimage.interpolation.rotate(img1, angle1, reshape=False, order=2)
        img2 = ndimage.interpolation.rotate(img2, angle2, reshape=False, order=2)
        flow = ndimage.interpolation.rotate(flow, angle1, reshape=False, order=2)

        target_ = np.copy(flow)
        flow[:, :, 0] = (
            np.cos(angle1_rad) * target_[:, :, 0]
            + np.sin(angle1_rad) * target_[:, :, 1]
        )
        flow[:, :, 1] = (
            -np.sin(angle1_rad) * target_[:, :, 0]
            + np.cos(angle1_rad) * target_[:, :, 1]
        )

    return img1, img2, flow


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
