import torch.nn as nn

from .units import ConvNormRelu


class DisplacementAwareProjection(nn.Module):
    """Displacement-aware projection layer"""

    def __init__(self, max_displacement=3, temperature=False, temp_factor=1e-6):
        super(DisplacementAwareProjection, self).__init__()

        self.temperature = temperature
        self.temp_factor = temp_factor

        dim_c = (2 * max_displacement + 1) ** 2

        if self.temperature:
            self.dap_layer = ConvNormRelu(
                dim_c, 1, kernel_size=1, padding=0, stride=1, norm=None, activation=None
            )

        else:
            self.dap_layer = ConvNormRelu(
                dim_c,
                dim_c,
                kernel_size=1,
                padding=0,
                stride=1,
                norm=None,
                activation=None,
            )

    def forward(self, x):

        x = x.squeeze(1)
        bs, du, dv, h, w = x.shape
        x = x.view(bs, du * dv, h, w)

        if self.temperature:
            temp = self.dap_layer(x) + self.temp_factor
            x = x * temp
        else:
            x = self.dap_layer(x)

        return x.view(bs, du, dv, h, w).unsqueeze(1)
