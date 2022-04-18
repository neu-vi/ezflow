import torch.nn as nn


class BaseModule(nn.Module):
    """
    A wrapper for torch.nn.Module to maintain common
    module functionalities.

    """

    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self):
        pass

    def freeze_batch_norm(self):
        """
        Set Batch Norm layers to evaluation state.
        This method can be used for fine tuning.

        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
