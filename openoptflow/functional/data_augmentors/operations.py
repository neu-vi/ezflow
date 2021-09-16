import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter


def color_transform(
    img1,
    img2,
    asymmetric_aug_prob=0.2,
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.5 / 3.14,
):

    aug = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    if np.random.rand() < asymmetric_aug_prob:
        img1 = np.array(aug(Image.fromarray(img1)), dtype=np.uint8)
        img2 = np.array(aug(Image.fromarray(img2)), dtype=np.uint8)

    else:
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

    return img1, img2


def eraser_transform(img1, img2, bounds=[50, 100], aug_prob=0.5):

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

    y0 = np.random.randint(0, img1.shape[0] - crop_size[0])
    x0 = np.random.randint(0, img1.shape[1] - crop_size[1])

    img1 = img1[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    img2 = img2[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
    flow = flow[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

    return img1, img2, flow
