import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter


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
        Probability of applying the augmentation
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
    crop_type="center",
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

    if crop_type.lower() == "center":
        y0 = int(H / 2 - crop_size[0] / 2)
        x0 = int(W / 2 - crop_size[1] / 2)

    else:
        y0 = np.random.randint(0, img1.shape[0] - crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - crop_size[1])

    img1 = img1[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    img2 = img2[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    flow = flow[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

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
    crop_type="center",
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

    margin_y = 20
    margin_x = 50

    if crop_type.lower() == "center":
        y0 = int(H / 2 - crop_size[0] / 2)
        x0 = int(W / 2 - crop_size[1] / 2)

    else:
        y0 = np.random.randint(0, img1.shape[0] - crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - crop_size[1] + margin_x)
        y0 = np.clip(y0, 0, img1.shape[0] - crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - crop_size[1])

    img1 = img1[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    img2 = img2[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    flow = flow[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    valid = valid[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    return img1, img2, flow, valid
