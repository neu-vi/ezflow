import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ..config import get_cfg
from ..data import DeviceDataLoader
from ..functional import FUNCTIONAL_REGISTRY
from ..utils import AverageMeter
from .metrics import endpointerror
from .registry import loss_functions, optimizers, schedulers


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader):

        self.cfg = cfg
        self.model_name = model.__class__.__name__.lower()
        device = cfg.DEVICE

        if isinstance(device, list) or isinstance(device, tuple):
            device = ",".join(map(str, device))

        print("\n")

        if device == "-1" or device == -1 or device == "cpu":
            device = torch.device("cpu")
            print("Running on CPU\n")

        elif not torch.cuda.is_available():
            device = torch.device("cpu")
            print("CUDA device(s) not available. Running on CPU\n")

        else:
            self.model_parallel = True

            if device == "all":
                device = torch.device("cuda")
                if cfg.DISTRIBUTED:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model)
                print(f"Running on all available CUDA devices\n")

            else:

                if type(device) != str:
                    device = str(device)

                device_ids = device.split(",")
                device_ids = [int(id) for id in device_ids]
                cuda_str = "cuda:" + device
                device = torch.device(cuda_str)
                if cfg.DISTRIBUTED:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model, device_ids=device_ids)
                print(f"Running on CUDA devices {device_ids}\n")

        self.device = device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _calculate_metric(self, pred, target):

        return endpointerror(pred, target)

    def _train_model(self, n_epochs, loss_fn, optimizer, scheduler):

        writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)

        best_model = deepcopy(self.model)

        model = self.model
        model.train()

        epoch_loss = AverageMeter()
        avg_metric = 0.0

        epochs = 0
        while epochs < n_epochs:

            print(f"Epoch {epochs+1} of {n_epochs}")
            print("-" * 80)

            epoch_loss.reset()
            for iteration, (inp, target) in enumerate(self.train_loader):

                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )

                pred = model(img1, img2)

                loss = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_loss.update(loss.item(), self.train_loader.batch_size)

                if iteration % self.cfg.LOG_ITERATIONS_INTERVAL == 0:

                    total_iters = iteration + (epochs * len(self.train_loader))
                    writer.add_scalar(
                        "avg_training_loss",
                        epoch_loss.avg,
                        total_iters,
                    )
                    print(
                        f"Epoch iterations: {iteration}, Total iterations:{total_iters}, Average training loss:{epoch_loss.avg}"
                    )

            if epochs % self.cfg.VAL_INTERVAL == 0:

                new_avg_metric = self._validate_model(model)
                if new_avg_metric > avg_metric:
                    best_model = deepcopy(model)
                    avg_metric = new_avg_metric

                writer.add_scalar("validation_metric", avg_metric, epochs + 1)
                print(f"Epoch {epochs}: Validation metric = {avg_metric}")

            print(f"Epoch {epochs}: Training loss = {epoch_loss.sum}")
            writer.add_scalar("epochs_training_loss", epoch_loss.sum, epochs + 1)

            if epochs % self.cfg.CKPT_INTERVAL == 0:

                if self.model_parallel:
                    save_model = best_model.module
                torch.save(
                    save_model.state_dict(),
                    os.path.join(
                        self.cfg.CKPT_DIR,
                        self.model_name + "_epochs" + str(epochs) + ".pth",
                    ),
                )

            epochs += 1
            print("\n")

        writer.close()

        return best_model

    def _validate_model(self, model):

        model.eval()

        metric_meter = AverageMeter()
        batch_size = self.val_loader.batch_size

        with torch.no_grad():
            for inp, target in self.val_loader:

                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )

                pred = model(img1, img2)

                metric = self._calculate_metric(pred, target)
                metric_meter.update(metric.item(), n=batch_size)

        return metric_meter.avg

    def train(self, n_epochs=None):

        if self.cfg.CRITERION.CUSTOM:
            loss_fn = FUNCTIONAL_REGISTRY.get(self.cfg.CRITERION.NAME)
        else:
            loss_fn = loss_functions.get(self.cfg.CRITERION.NAME)

        if self.cfg.CRITERION.PARAMS:
            loss_params = self.cfg.CRITERION.PARAMS.to_dict()
            loss_fn = loss_fn(**loss_params)
        else:
            loss_fn = loss_fn()

        optimizer = optimizers.get(self.cfg.OPTIMIZER.NAME)

        if self.cfg.OPTIMIZER.PARAMS:
            optimizer_params = self.cfg.OPTIMIZER.PARAMS.to_dict()
            optimizer = optimizer(self.model.parameters(), **optimizer_params)
        else:
            optimizer = optimizer(self.model.parameters())

        scheduler = None
        if self.cfg.SCHEDULER.USE:
            scheduler = schedulers.get(self.cfg.SCHEDULER.NAME)

            if self.cfg.SCHEDULER.PARAMS:
                scheduler_params = self.cfg.SCHEDULER.PARAMS.to_dict()
                scheduler = scheduler(optimizer, **scheduler_params)
            else:
                scheduler = scheduler(optimizer)

        if n_epochs is None:
            n_epochs = self.cfg.EPOCHS

        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)

        print(f"Training {self.model_name.upper()} for {n_epochs} epochs\n")
        best_model = self._train_model(n_epochs, loss_fn, optimizer, scheduler)
        print("Training complete!")

        if self.model_parallel:
            best_model = best_model.module

        torch.save(
            best_model.state_dict(),
            os.path.join(self.cfg.CKPT_DIR, self.model_name + "_best.pth"),
        )
        print("Saved best model!\n")

    def validate(self, model=None):

        if model is None:
            model = self.model

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            metric = self._validate_model(model)

        print(f"Average validation metric = {metric}")

        return metric


def get_training_cfg(cfg_path, custom=True):

    """
    Parameters
    ----------
    cfg_path : str
        Path to the config file.
    custom : bool
        If True, the config file is assumed to be a custom config file.
        If False, the config file is assumed to be a standard config file present in openoptflow/configs/trainers.

    Returns
    -------
    cfg : Config
        The config object.
    """

    return get_cfg(cfg_path, custom=custom, grp="trainers")
