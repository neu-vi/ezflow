import os
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ..functional import FUNCTIONAL_REGISTRY
from ..utils import AverageMeter, endpointerror
from .registry import loss_functions, optimizers, schedulers


def seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Trainer:
    """Trainer class for training and evaluating models

    Parameters
    ----------

    cfg : CfgNode
        Configuration object for training
    model : torch.nn.Module
        Model to be trained
    train_loader : torch.utils.data.DataLoader
        DataLoader for training
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation
    """

    def __init__(self, cfg, model, train_loader, val_loader):

        self.cfg = cfg

        self.model = model
        self.model_name = model.__class__.__name__.lower()
        self._setup_model(model)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _setup_model(self, model):

        device = self.cfg.DEVICE

        if isinstance(device, list) or isinstance(device, tuple):
            device = ",".join(map(str, device))

        print("\n")

        self.model_parallel = False

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

                if self.cfg.DISTRIBUTED.USE is True:
                    self._setup_ddp()
                    model = DDP(model.cuda(), device_ids=[self.cfg.DISTRIBUTED.RANK])
                else:
                    model = nn.DataParallel(model)

                print(f"Running on all available CUDA devices\n")

            else:

                if type(device) != str:
                    device = str(device)

                device_ids = device.split(",")
                device_ids = [int(id) for id in device_ids]
                device = torch.device("cuda")

                if self.cfg.DISTRIBUTED.USE is True:
                    self._setup_ddp()
                    model = DDP(model.cuda(), device_ids=[self.cfg.DISTRIBUTED.RANK])
                    print("Performing distributed training")
                else:
                    model = nn.DataParallel(model, device_ids=device_ids)

                print(f"Running on CUDA devices {device_ids}\n")

        self.device = device
        self.model = model.to(self.device)

    def _setup_ddp(self):

        os.environ["MASTER_ADDR"] = self.cfg.DISTRIBUTED.MASTER_ADDR
        os.environ["MASTER_PORT"] = self.cfg.DISTRIBUTED.MASTER_PORT

        seed(0)

        dist.init_process_group(
            backend=self.cfg.DISTRIBUTED.BACKEND,
            world_size=self.cfg.DISTRIBUTED.WORLD_SIZE,
            rank=self.cfg.DISTRIBUTED.RANK,
        )

    def _setup_training(self, loss_fn=None, optimizer=None, scheduler=None):

        if loss_fn is None:

            if self.cfg.CRITERION.CUSTOM:
                loss = FUNCTIONAL_REGISTRY.get(self.cfg.CRITERION.NAME)
            else:
                loss = loss_functions.get(self.cfg.CRITERION.NAME)

            if self.cfg.CRITERION.PARAMS is not None:
                loss_params = self.cfg.CRITERION.PARAMS.to_dict()
                loss_fn = loss(**loss_params)
            else:
                loss_fn = loss()

        if optimizer is None:

            opt = optimizers.get(self.cfg.OPTIMIZER.NAME)

            if self.cfg.OPTIMIZER.PARAMS is not None:
                optimizer_params = self.cfg.OPTIMIZER.PARAMS.to_dict()
                optimizer = opt(
                    self.model.parameters(),
                    lr=self.cfg.OPTIMIZER.LR,
                    **optimizer_params,
                )
            else:
                optimizer = opt(self.model.parameters(), lr=self.cfg.OPTIMIZER.LR)

        if scheduler is None:

            if self.cfg.SCHEDULER.USE:
                sched = schedulers.get(self.cfg.SCHEDULER.NAME)

                if self.cfg.SCHEDULER.PARAMS is not None:
                    scheduler_params = self.cfg.SCHEDULER.PARAMS.to_dict()
                    scheduler = sched(optimizer, **scheduler_params)
                else:
                    scheduler = sched(optimizer)

        return loss_fn, optimizer, scheduler

    def _calculate_metric(self, pred, target):

        return endpointerror(pred, target)

    def _train_model(self, loss_fn, optimizer, scheduler, n_epochs, start_epoch=None):

        writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)

        model = self.model
        best_model = deepcopy(model)
        model.train()

        self.loss_fn = loss_fn

        epoch_loss = AverageMeter()
        min_avg_val_loss = float("inf")
        min_avg_val_metric = float("inf")

        if start_epoch is not None:
            print(f"Resuming training from epoch {start_epoch+1}\n")
        else:
            start_epoch = 0

        for epochs in range(start_epoch, start_epoch + n_epochs):

            print(f"Epoch {epochs+1} of {start_epoch+n_epochs}")
            print("-" * 80)

            epoch_loss.reset()
            for iteration, (inp, target) in enumerate(self.train_loader):

                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )
                target = target / self.cfg.DATA.TARGET_SCALE_FACTOR

                pred = model(img1, img2)

                loss = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()

                if self.cfg.GRAD_CLIP.USE is True:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.GRAD_CLIP.VALUE
                    )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_loss.update(loss.item())

                if iteration % self.cfg.LOG_ITERATIONS_INTERVAL == 0:

                    total_iters = iteration + (epochs * len(self.train_loader))
                    writer.add_scalar(
                        "avg_batch_training_loss",
                        epoch_loss.avg,
                        total_iters,
                    )
                    print(
                        f"Epoch iterations: {iteration}, Total iterations: {total_iters}, Average batch training loss: {epoch_loss.avg}"
                    )

            print(f"\nEpoch {epochs+1}: Training loss = {epoch_loss.sum}")
            writer.add_scalar("epochs_training_loss", epoch_loss.sum, epochs + 1)

            if epochs % self.cfg.VALIDATE_INTERVAL == 0:

                new_avg_val_loss, new_avg_val_metric = self._validate_model(model)

                writer.add_scalar("avg_validation_loss", new_avg_val_loss, epochs + 1)
                print(f"Epoch {epochs+1}: Average validation loss = {new_avg_val_loss}")

                writer.add_scalar(
                    "avg_validation_metric", new_avg_val_metric, epochs + 1
                )
                print(
                    f"Epoch {epochs+1}: Average validation metric = {new_avg_val_metric}"
                )

                if new_avg_val_loss < min_avg_val_loss:

                    min_avg_val_loss = new_avg_val_loss
                    print("New minimum average validation loss!")

                    if self.cfg.VALIDATE_ON.lower() == "loss":
                        best_model = deepcopy(model)
                        save_best_model = (
                            best_model.module if self.model_parallel else best_model
                        )
                        torch.save(
                            save_best_model.state_dict(),
                            os.path.join(
                                self.cfg.CKPT_DIR, self.model_name + "_best.pth"
                            ),
                        )
                        print(f"Saved new best model at epoch {epochs+1}!")

                if new_avg_val_metric < min_avg_val_metric:

                    min_avg_val_metric = new_avg_val_metric
                    print("New minimum average validation metric!")

                    if self.cfg.VALIDATE_ON.lower() == "metric":
                        best_model = deepcopy(model)
                        save_best_model = (
                            best_model.module if self.model_parallel else best_model
                        )
                        torch.save(
                            save_best_model.state_dict(),
                            os.path.join(
                                self.cfg.CKPT_DIR, self.model_name + "_best.pth"
                            ),
                        )
                        print(f"Saved new best model at epoch {epochs+1}!")

            if epochs % self.cfg.CKPT_INTERVAL == 0:

                if self.model_parallel:
                    save_model = model.module
                else:
                    save_model = model

                consolidated_save_dict = {
                    "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": epochs,
                }
                if scheduler is not None:
                    consolidated_save_dict[
                        "scheduler_state_dict"
                    ] = scheduler.state_dict()

                torch.save(
                    consolidated_save_dict,
                    os.path.join(
                        self.cfg.CKPT_DIR,
                        self.model_name + "_epochs" + str(epochs + 1) + ".pth",
                    ),
                )

            print("\n")

        writer.close()

        return best_model

    def _validate_model(self, model):

        model.eval()

        metric_meter = AverageMeter()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for inp, target in self.val_loader:

                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )
                target = target / self.cfg.DATA.TARGET_SCALE_FACTOR

                pred = model(img1, img2)

                loss = self.loss_fn(pred, target)
                loss_meter.update(loss.item())

                metric = self._calculate_metric(pred, target)
                metric_meter.update(metric.item())

        model.train()

        return loss_meter.avg, metric_meter.avg

    def train(
        self,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        start_epoch=None,
    ):
        """
        Method to train the model

        Parameters
        ----------
        loss_fn : torch.nn.modules.loss, optional
            The loss function to be used. Defaults to None (which uses the loss function specified in the config file).
        optimizer : torch.optim.Optimizer, optional
            The optimizer to be used. Defaults to None (which uses the optimizer specified in the config file).
        scheduler : torch.optim.lr_scheduler, optional
            The learning rate scheduler to be used. Defaults to None (which uses the scheduler specified in the config file).
        n_epochs : int, optional
            The number of epochs to train for. Defaults to None (which uses the number of epochs specified in the config file).
        start_epoch : int, optional
            The epoch to start training from. Defaults to None (which starts from 0).

        """

        loss_fn, optimizer, scheduler = self._setup_training(
            loss_fn, optimizer, scheduler
        )

        if n_epochs is None:
            n_epochs = self.cfg.EPOCHS

        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)

        print(f"Training {self.model_name.upper()} for {n_epochs} epochs\n")
        best_model = self._train_model(
            loss_fn, optimizer, scheduler, n_epochs, start_epoch
        )
        print("Training complete!")

        if self.model_parallel:
            best_model = best_model.module

        torch.save(
            best_model.state_dict(),
            os.path.join(self.cfg.CKPT_DIR, self.model_name + "_best_final.pth"),
        )
        print("Saved best model!\n")

    def train_distributed(
        self,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        start_epoch=None,
    ):
        """
        Method to train the model in a distributed fashion using DDP

        Parameters
        ----------
        loss_fn : torch.nn.modules.loss, optional
            The loss function to be used. Defaults to None (which uses the loss function specified in the config file).
        optimizer : torch.optim.Optimizer, optional
            The optimizer to be used. Defaults to None (which uses the optimizer specified in the config file).
        scheduler : torch.optim.lr_scheduler, optional
            The learning rate scheduler to be used. Defaults to None (which uses the scheduler specified in the config file).
        n_epochs : int, optional
            The number of epochs to train for. Defaults to None (which uses the number of epochs specified in the config file).
        start_epoch : int, optional
            The epoch to start training from. Defaults to None (which starts from 0).

        """

        mp.spawn(
            self.train,
            args=(loss_fn, optimizer, scheduler, n_epochs, start_epoch),
            nprocs=self.cfg.DISTRIBUTED.WORLD_SIZE,
        )

    def resume_training(
        self,
        consolidated_ckpt=None,
        model_ckpt=None,
        optimizer_ckpt=None,
        n_epochs=None,
        start_epoch=None,
        scheduler_ckpt=None,
        use_cfg=False,
    ):

        """
        Method to resume training of a model

        Parameters
        ----------
        consolidated_ckpt : str, optional
            The path to the consolidated checkpoint file. Defaults to None (which uses the consolidated checkpoint file specified in the config file).
        model_ckpt : str, optional
            The path to the model checkpoint file. Defaults to None (which uses the model checkpoint file specified in the config file).
        optimizer_ckpt : str, optional
            The path to the optimizer checkpoint file. Defaults to None (which uses the optimizer checkpoint file specified in the config file).
        n_epochs : int, optional
            The number of epochs to train for. Defaults to None (which uses the number of epochs specified in the config file).
        start_epoch : int, optional
            The epoch to start training from. Defaults to None (which infers the last epoch from the ckpt).
        scheduler_ckpt : str, optional
            The path to the scheduler checkpoint file. Defaults to None (which uses the scheduler checkpoint file specified in the config file).
        use_cfg : bool, optional
            Whether to use the config file or not. Defaults to False.

        """

        consolidated_ckpt = (
            self.cfg.RESUME_TRAINING.CONSOLIDATED_CKPT
            if use_cfg is True
            else consolidated_ckpt
        )

        if consolidated_ckpt is not None:

            ckpt = torch.load(consolidated_ckpt, map_location=torch.device("cpu"))

            model_state_dict = ckpt["model_state_dict"]
            optimizer_state_dict = ckpt["optimizer_state_dict"]

            if "scheduler_state_dict" in ckpt.keys():
                scheduler_state_dict = ckpt["scheduler_state_dict"]

            if "epochs" in ckpt.keys():
                start_epoch = ckpt["epochs"] + 1

        else:

            assert (
                model_ckpt is not None and optimizer_ckpt is not None
            ), "Must provide a consolidated ckpt or model and optimizer ckpts separately"

            model_state_dict = torch.load(model_ckpt, map_location=torch.device("cpu"))
            optimizer_state_dict = torch.load(
                optimizer_ckpt, map_location=torch.device("cpu")
            )

            if scheduler_ckpt is not None:
                scheduler_state_dict = torch.load(
                    scheduler_ckpt, map_location=torch.device("cpu")
                )

        if self.model_parallel:
            model = self.model.module
        else:
            model = self.model

        model.load_state_dict(model_state_dict)
        self._setup_model(model)

        loss_fn, optimizer, scheduler = self._setup_training()
        optimizer.load_state_dict(optimizer_state_dict)

        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state_dict)

        if n_epochs is None and use_cfg:
            n_epochs = self.cfg.RESUME_TRAINING.EPOCHS
        if start_epoch is None and use_cfg:
            start_epoch = self.cfg.RESUME_TRAINING.START_EPOCH

        self.train(loss_fn, optimizer, scheduler, n_epochs, start_epoch)

    def resume_distributed_training(
        self,
        consolidated_ckpt=None,
        model_ckpt=None,
        optimizer_ckpt=None,
        n_epochs=None,
        start_epoch=None,
        scheduler_ckpt=None,
        use_cfg=False,
    ):

        """
        Method to resume training of a model in a distributed fashion using DDP

        Parameters
        ----------
        consolidated_ckpt : str, optional
            The path to the consolidated checkpoint file. Defaults to None (which uses the consolidated checkpoint file specified in the config file).
        model_ckpt : str, optional
            The path to the model checkpoint file. Defaults to None (which uses the model checkpoint file specified in the config file).
        optimizer_ckpt : str, optional
            The path to the optimizer checkpoint file. Defaults to None (which uses the optimizer checkpoint file specified in the config file).
        n_epochs : int, optional
            The number of epochs to train for. Defaults to None (which uses the number of epochs specified in the config file).
        start_epoch : int, optional
            The epoch to start training from. Defaults to None (which infers the last epoch from the ckpt).
        scheduler_ckpt : str, optional
            The path to the scheduler checkpoint file. Defaults to None (which uses the scheduler checkpoint file specified in the config file).
        use_cfg : bool, optional
            Whether to use the config file or not. Defaults to False.

        """

        mp.spawn(
            self.resume_training,
            args=(
                consolidated_ckpt,
                model_ckpt,
                optimizer_ckpt,
                n_epochs,
                start_epoch,
                scheduler_ckpt,
                use_cfg,
            ),
            nprocs=self.cfg.DISTRIBUTED.WORLD_SIZE,
        )

    def validate(self, model=None):

        """
        Method to validate the model

        Parameters
        ----------
        model : torch.nn.Module, optional
            The model to be used for validation. Defaults to None (which uses the model which was trained).
        """

        if model is None:
            model = self.model

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            avg_val_loss, avg_val_metric = self._validate_model(model)

        print(f"Average validation loss = {avg_val_loss}")
        print(f"Average validation metric = {avg_val_metric}")

        return avg_val_loss, avg_val_metric
