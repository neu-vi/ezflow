import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    """
    Basic residual block for ResNet-style architectures
    """

    def __init__(self, in_channels, out_channels, stride=1, norm="group"):
        super(BasicBlock, self).__init__()

        assert norm in ("group", "batch", "instance", "none")

        if norm == "group":
            n_groups = out_channels // 8
            norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
            norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)

            if stride != 1:
                norm3 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)

        elif norm == "batch":
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)

            if stride != 1:
                norm3 = nn.BatchNorm2d(out_channels)

        elif norm == "instance":
            norm1 = nn.InstanceNorm2d(out_channels)
            norm2 = nn.InstanceNorm2d(out_channels)

            if stride != 1:
                norm3 = nn.InstanceNorm2d(out_channels)

        elif norm == "none":
            norm1 = nn.Sequential()
            norm2 = nn.Sequential()

            if stride != 1:
                norm3 = nn.Sequential()

        self.residual_fn = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, stride=stride
            ),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm2,
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm3,
            )

    def forward(self, x):

        out = self.residual_fn(x)
        out = F.relu(out + self.shortcut(x), inplace=True)

        return out


class BottleneckBlock(nn.Module):

    """
    Bottleneck residual block for ResNet-style architectures
    """

    def __init__(self, in_channels, out_channels, stride=1, norm="group"):
        super(BottleneckBlock, self).__init__()

        assert norm in ("group", "batch", "instance", "none")

        self.in_channels = in_channels
        self.out_channels = out_channels

        if norm == "group":
            num_groups = out_channels // 8
            norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels // 4)
            norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels // 4)
            norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

            if not stride == 1:
                norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        elif norm == "batch":
            norm1 = nn.BatchNorm2d(out_channels // 4)
            norm2 = nn.BatchNorm2d(out_channels // 4)
            norm3 = nn.BatchNorm2d(out_channels)

            if not stride == 1:
                norm4 = nn.BatchNorm2d(out_channels)

        elif norm == "instance":
            norm1 = nn.InstanceNorm2d(out_channels // 4)
            norm2 = nn.InstanceNorm2d(out_channels // 4)
            norm3 = nn.InstanceNorm2d(out_channels)

            if not stride == 1:
                norm4 = nn.InstanceNorm2d(out_channels)

        elif norm == "none":
            norm1 = nn.Sequential()
            norm2 = nn.Sequential()
            norm3 = nn.Sequential()

            if not stride == 1:
                norm4 = nn.Sequential()

        self.residual_fn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 4,
                out_channels // 4,
                stride=stride,
                kernel_size=3,
                padding=1,
            ),
            norm2,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            norm3,
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm4,
            )

    def forward(self, x):

        out = self.residual_fn(x)
        out = F.relu(out + self.shortcut(x), inplace=True)

        return out
