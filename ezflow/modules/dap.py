import torch.nn as nn

from ..config import configurable
from .build import MODULE_REGISTRY
from .units import ConvNormRelu


@MODULE_REGISTRY.register()
class DisplacementAwareProjection(nn.Module):
    """
    Displacement-aware projection layer

    Parameters
    ----------
    max_displacement : int, optional
        Maximum displacement
    temperature : bool, optional
        If True, use temperature scaling
    temp_factor : float, optional
        Temperature scaling factor
    """

    @configurable
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

    @classmethod
    def from_config(cls, cfg):
        return {
            "max_displacement": cfg.MAX_DISPLACEMENT,
            "temperature": cfg.TEMPERATURE,
            "temp_factor": cfg.TEMP_FACTOR,
        }

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
