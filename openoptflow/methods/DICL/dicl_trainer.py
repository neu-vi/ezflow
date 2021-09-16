import torch.nn as nn
import torch.nn.functional as F

from ...base_trainer import BaseTrainer
from ...functional import MultiScaleLoss


class DICLTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        save_dir=".",
        save_interval=5,
        log_dir=".",
        device="cpu",
        **kwargs
    ):
        super(DICLTrainer, self).__init__(
            model,
            train_loader,
            val_loader,
            optimizer,
            save_dir,
            save_interval,
            log_dir,
            device,
        )

        self.loss_fn = MultiScaleLoss(**kwargs)

    def _calculate_loss(self, pred, label):

        return self.loss_fn(pred, label)
