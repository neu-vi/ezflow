import torch
import torch.nn as nn
import torch.nn.functional as F

from .....models import ASPPModule


class UNetGroupsASPP(nn.Module):
    def __init__(
        self,
        in_planes,
        spatial_dim,
        mid_planes,
        out_planes,
        inter_planes,
        nln_pos,
        use_cbam,
        cbam_reduction,
        cbam_spatial,
        use_se,
        num_dilations,
        use_heavy_stem,
    ):
        super(UNetGroupsASPP, self).__init__()

        self.pre_stem = nn.Sequential(
            nn.Conv3d(
                in_planes,
                in_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_dilations,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_planes,
                in_planes,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_dilations,
            ),
            nn.ReLU(),
        )

        if use_heavy_stem:
            self.stem = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    inter_planes,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                    stride=(1, 1, 1),
                ),
                nn.ReLU(),
                nn.Conv3d(
                    inter_planes,
                    inter_planes,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                    stride=(1, 1, 1),
                ),
                nn.ReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    inter_planes,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                    stride=(1, 1, 1),
                ),
                nn.ReLU(),
            )

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                inter_planes, inter_planes * 2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
            nn.Conv3d(
                inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
            nn.Conv3d(
                inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
            ASPPModule(
                inter_planes * 2,
                2 * inter_planes,
                inter_planes * 2,
                dilations=(2, 4, 8),
            ),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                inter_planes * 2 + inter_planes * 2,
                inter_planes * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                inter_planes * 2, inter_planes * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(
                inter_planes * 2 + inter_planes,
                inter_planes,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv3d(inter_planes, inter_planes, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.flow_out = nn.Conv3d(inter_planes, out_planes, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.pre_stem(x)
        x = self.stem(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, size=x1.shape[2:], mode="trilinear", align_corners=False)
        x3 = torch.cat((x3, x1), dim=1)
        x4 = self.conv4(x3)
        x4 = F.interpolate(x4, size=x.shape[2:], mode="trilinear", align_corners=False)
        x4 = torch.cat((x4, x), dim=1)
        x5 = self.conv5(x4)
        out = self.flow_out(x5)

        return out
