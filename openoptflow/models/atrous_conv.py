import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):

    """
    Atrous Spatial Pyramid Pooling module from the paper "Rethinking Atrous Convolution for Semantic Image Segmentation"
    """

    def __init__(
        self, features, inner_features=512, out_features=512, dilations=(12, 24, 36)
    ):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                features,
                inner_features,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                inner_features * 5,
                out_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=False
        )

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)

        return bottle
