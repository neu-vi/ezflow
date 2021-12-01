import numpy as np
import torch.optim as optim

from .registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Coaine learning rate warmup scheduler

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to be used with the scheduler
    warmup : int
        Number of epochs to warmup the learning rate
    max_iters : int
        Maximum number of iterations to train the model
    """

    def __init__(self, optimizer, warmup=100, max_iters=200):
        super().__init__(optimizer)

        self.warmup = warmup
        self.max_num_iters = max_iters

    def get_lr(self):

        lr_factor = self.get_lr_factor(epoch=self.last_epoch)

        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Parameters
        ----------
        epoch : int
            Current epoch

        Returns
        -------
        float
            Learning rate factor
        """

        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup

        return lr_factor
