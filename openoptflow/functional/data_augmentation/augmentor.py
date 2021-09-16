from .operations import *


class FlowAugmentor:
    def __init__(
        self,
        crop_size,
        color_aug_params={"aug_prob": 0.2},
        eraser_aug_params={"aug_prob": 0.5},
        spatial_aug_params={"aug_prob": 0.8},
    ):

        self.crop_size = crop_size
        self.color_aug_params = color_aug_params
        self.eraser_aug_params = eraser_aug_params
        self.spatial_aug_params = spatial_aug_params

    def __call__(self, img1, img2, flow):

        img1, img2 = color_transform(img1, img2, **self.color_aug_params)
        img1, img2 = eraser_transform(img1, img2, **self.eraser_aug_params)
        img1, img2, flow = spatial_transform(
            img1, img2, flow, self.crop_size, **self.spatial_aug_params
        )

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow
