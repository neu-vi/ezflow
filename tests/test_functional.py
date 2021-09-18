import numpy as np

from openoptflow.functional import FlowAugmentor

img1 = np.random.rand(256, 256, 3).astype(np.uint8)
img2 = np.random.rand(256, 256, 3).astype(np.uint8)
flow = np.random.rand(256, 256, 2).astype(np.float32)


def test_FlowAugmentor():

    augmentor = FlowAugmentor((224, 224))
    _ = augmentor(img1, img2, flow)
