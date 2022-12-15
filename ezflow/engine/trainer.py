import os
import time
from copy import deepcopy
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ..functional import FUNCTIONAL_REGISTRY
from ..utils import AverageMeter, endpointerror, find_free_port, is_port_available
from .registry import loss_functions, optimizers, schedulers


def seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class BaseTrainer:
    def __init__(self):
        self.cfg = None

        self.model = None

        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        self.train_loader = None
        self.val_loader = None

        self.device = None
        self._trainer = None

        self.model_parallel = False

        self.writer = None
        self.times = []

    def _setup_device(self):
        raise NotImplementedError

    def _setup_model(self):
        raise NotImplementedError

    def _is_main_process(self):
        raise NotImplementedError

    def _setup_training(self, rank=0, loss_fn=None, optimizer=None, scheduler=None):
        if loss_fn is None and self.loss_fn is None:

            if self.cfg.CRITERION.CUSTOM:
                loss = FUNCTIONAL_REGISTRY.get(self.cfg.CRITERION.NAME)
            else:
                loss = loss_functions.get(self.cfg.CRITERION.NAME)

            if self.cfg.CRITERION.PARAMS is not None:
                loss_params = self.cfg.CRITERION.PARAMS.to_dict()
                loss_fn = loss(**loss_params)
            else:
                loss_fn = loss()

            print(f"Loss function: {self.cfg.CRITERION.NAME} is initialized!")

        if optimizer is None and self.optimizer is None:

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

            print(f"Optimizer: {self.cfg.OPTIMIZER.NAME} is initialized!")

        if scheduler is None and self.scheduler is None:

            if self.cfg.SCHEDULER.USE:
                sched = schedulers.get(self.cfg.SCHEDULER.NAME)

                if self.cfg.SCHEDULER.PARAMS is not None:
                    scheduler_params = self.cfg.SCHEDULER.PARAMS.to_dict()
                    if "epochs" in scheduler_params:
                        scheduler_params["steps_per_epoch"] = len(self.train_loader)

                    scheduler = sched(optimizer, **scheduler_params)
                else:
                    scheduler = sched(optimizer)

                print(f"Scheduler: {self.cfg.SCHEDULER.NAME} is initialized!")

        if self.loss_fn is None:
            self.loss_fn = loss_fn

        if self.optimizer is None:
            self.optimizer = optimizer

        if self.scheduler is None:
            self.scheduler = scheduler

        if rank == 0:
            """
            Initialize Tensorboard SummyWriter only for main process
            """
            self.writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)

        self.scaler = GradScaler(enabled=self.cfg.MIXED_PRECISION)

        self._trainer = (
            self._step_trainer
            if self.cfg.NUM_STEPS is not None
            else self._epoch_trainer
        )

        self.min_avg_val_loss = float("inf")
        self.min_avg_val_metric = float("inf")

    def _freeze_bn(self):
        if self.cfg.FREEZE_BATCH_NORM:
            if self.model_parallel:
                self.model.module.freeze_batch_norm()
            else:
                self.model.freeze_batch_norm()

    def _epoch_trainer(self, n_epochs=None, start_epoch=None):
        self.model.train()
        self._freeze_bn()

        loss_meter = AverageMeter()

        if n_epochs is None:
            n_epochs = self.cfg.EPOCHS

        if start_epoch is not None:
            print(f"Resuming training from epoch {start_epoch+1}\n")
        else:
            start_epoch = 0

        for epoch in range(start_epoch, start_epoch + n_epochs):

            print(f"\nEpoch {epoch+1} of {start_epoch+n_epochs}")
            print("-" * 80)

            if self.model_parallel:
                self.train_loader.sampler.set_epoch(epoch)

            loss_meter.reset()
            for iteration, (inp, target) in enumerate(self.train_loader):

                loss = self._run_step(inp, target)

                loss_meter.update(loss.item())

                total_iters = iteration + (epoch * len(self.train_loader))
                self._log_step(iteration, total_iters, loss_meter)

            print(f"\nEpoch {epoch+1}: Training loss = {loss_meter.sum}")

            if self._is_main_process():
                self.writer.add_scalar(
                    "epochs_training_loss", loss_meter.sum, epoch + 1
                )

            if epoch % self.cfg.VALIDATE_INTERVAL == 0 and self._is_main_process():
                self._validate_model(iter_type="Epoch", iterations=epoch + 1)

            if epoch % self.cfg.CKPT_INTERVAL == 0 and self._is_main_process():
                self._save_checkpoints(ckpt_type="epoch", ckpt_number=epoch + 1)

            # Synchronize all processes in multi gpu after validation and checkpoint
            if (
                epoch % self.cfg.VALIDATE_INTERVAL == 0
                or epoch % self.cfg.CKPT_INTERVAL == 0
            ) and self.model_parallel:
                dist.barrier()

        if self._is_main_process():
            self.writer.close()

    def _step_trainer(self, n_steps=None, start_step=None):
        self.model.train()
        self._freeze_bn()

        loss_meter = AverageMeter()

        total_steps = 0

        if n_steps is None:
            n_steps = self.cfg.NUM_STEPS

        if start_step is not None:
            print(f"Resuming training from step {start_step}\n")
            total_steps = start_step
            n_steps += start_step - 1
        else:
            start_step = total_steps = 1
            n_steps += start_step

        if self.model_parallel:
            epoch = 0
            self.train_loader.sampler.set_epoch(epoch)

        train_iter = iter(self.train_loader)

        print(f"\nStarting step {total_steps} of {n_steps}")
        print("-" * 80)
        for step in range(start_step, n_steps):
            try:
                inp, target = next(train_iter)
            except:
                if self.model_parallel:
                    epoch += 1
                    self.train_loader.sampler.set_epoch(epoch)

                # Handle exception if there is no data
                # left in train iterator to continue training.
                train_iter = iter(self.train_loader)
                inp, target = next(train_iter)

            loss = self._run_step(inp, target)
            loss_meter.update(loss.item())

            self._log_step(step, total_steps, loss_meter)

            if step % self.cfg.VALIDATE_INTERVAL == 0 and self._is_main_process():
                self._validate_model(iter_type="Iteration", iterations=total_steps)
                print("-" * 80)

            if step % self.cfg.CKPT_INTERVAL == 0 and self._is_main_process():
                self._save_checkpoints(ckpt_type="step", ckpt_number=total_steps)

            # Synchronize all processes in multi gpu after validation and checkpoint
            if (
                step % self.cfg.VALIDATE_INTERVAL == 0
                or step % self.cfg.CKPT_INTERVAL == 0
            ) and self.model_parallel:
                dist.barrier()

            total_steps += 1

        if self._is_main_process():
            self.writer.close()

    def _run_step(self, inp, target):
        img1, img2 = inp
        img1, img2, target = (
            img1.to(self.device),
            img2.to(self.device),
            target.to(self.device),
        )

        if self._is_main_process():
            start_time = time.time()

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            output = self.model(img1, img2)
            loss = self.loss_fn(
                output["flow_preds"], target / self.cfg.TARGET_SCALE_FACTOR
            )

            del output

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        if self.cfg.GRAD_CLIP.USE is True:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP.VALUE)

        self.scaler.step(self.optimizer)

        if self.scheduler is not None:
            self.scheduler.step()

        self.scaler.update()

        if self._is_main_process():
            self.times.append(time.time() - start_time)

        return loss

    def _log_step(self, iteration, total_iters, loss_meter):
        if iteration % self.cfg.LOG_ITERATIONS_INTERVAL == 0:
            print(
                f"Iterations: {iteration}, Total iterations: {total_iters}, Average batch training loss: {loss_meter.avg}"
            )
            if self._is_main_process():
                self.writer.add_scalar(
                    "avg_batch_training_loss",
                    loss_meter.avg,
                    total_iters,
                )

    def _validate_model(self, iter_type, iterations):
        self.model.eval()
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

                if self.model_parallel:
                    output = self.model.module(img1, img2)
                else:
                    output = self.model(img1, img2)

                loss = self.loss_fn(
                    output["flow_preds"], target / self.cfg.TARGET_SCALE_FACTOR
                )

                loss_meter.update(loss.item())

                """
                    Predicted upsampled flow should be scaled for EPE calculation.
                """
                metric = self._calculate_metric(
                    output["flow_upsampled"] * self.cfg.TARGET_SCALE_FACTOR, target
                )
                metric_meter.update(metric)

                del output

        new_avg_val_loss, new_avg_val_metric = loss_meter.avg, metric_meter.avg

        print("\n", "-" * 80)
        self.writer.add_scalar("avg_validation_loss", new_avg_val_loss, iterations)
        self.writer.add_scalar("avg_validation_metric", new_avg_val_metric, iterations)

        print(
            f"\n{iter_type} {iterations}: Average validation loss = {new_avg_val_loss}"
        )

        print(
            f"{iter_type} {iterations}: Average validation metric = {new_avg_val_metric}\n"
        )
        print("-" * 80, "\n")

        self._save_best_model(new_avg_val_loss, new_avg_val_metric)

        self.model.train()
        self._freeze_bn()

    def _calculate_metric(self, pred, target):
        return endpointerror(pred, target)

    def _save_checkpoints(self, ckpt_type, ckpt_number):
        if self.model_parallel:
            save_model = self.model.module
        else:
            save_model = self.model

        consolidated_save_dict = {
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            ckpt_type: ckpt_number,
        }
        if self.scheduler is not None:
            consolidated_save_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(
            consolidated_save_dict,
            os.path.join(
                self.cfg.CKPT_DIR,
                self.model_name + "_" + ckpt_type + str(ckpt_number) + ".pth",
            ),
        )

    def _save_best_model(self, new_avg_val_loss, new_avg_val_metric):
        if new_avg_val_loss < self.min_avg_val_loss:

            self.min_avg_val_loss = new_avg_val_loss
            print("\nNew minimum average validation loss!")

            if self.cfg.VALIDATE_ON.lower() == "loss":
                best_model = deepcopy(self.model)
                save_best_model = (
                    best_model.module if self.model_parallel else best_model
                )
                torch.save(
                    save_best_model.state_dict(),
                    os.path.join(self.cfg.CKPT_DIR, self.model_name + "_best.pth"),
                )
                print(f"Saved new best model!\n")

        if new_avg_val_metric < self.min_avg_val_metric:

            self.min_avg_val_metric = new_avg_val_metric
            print("\nNew minimum average validation metric!")

            if self.cfg.VALIDATE_ON.lower() == "metric":
                best_model = deepcopy(self.model)
                save_best_model = (
                    best_model.module if self.model_parallel else best_model
                )
                torch.save(
                    save_best_model.state_dict(),
                    os.path.join(self.cfg.CKPT_DIR, self.model_name + "_best.pth"),
                )
                print(f"Saved new best model!\n")

    def _reload_trainer_states(
        self,
        consolidated_ckpt=None,
        model_ckpt=None,
        optimizer_ckpt=None,
        total_iterations=None,
        start_iteration=None,
        scheduler_ckpt=None,
        use_cfg=False,
    ):

        self._setup_device()

        consolidated_ckpt = (
            self.cfg.RESUME_TRAINING.CONSOLIDATED_CKPT
            if use_cfg is True
            else consolidated_ckpt
        )

        if consolidated_ckpt is not None:

            ckpt = torch.load(consolidated_ckpt, map_location=self.device)

            model_state_dict = ckpt["model_state_dict"]
            optimizer_state_dict = ckpt["optimizer_state_dict"]

            if "scheduler_state_dict" in ckpt.keys():
                scheduler_state_dict = ckpt["scheduler_state_dict"]

            if "epochs" in ckpt.keys():
                start_iteration = ckpt["epochs"] + 1

            if "step" in ckpt.keys():
                start_iteration = ckpt["step"] + 1

        else:

            assert (
                model_ckpt is not None and optimizer_ckpt is not None
            ), "Must provide a consolidated ckpt or model and optimizer ckpts separately"

            model_state_dict = torch.load(model_ckpt, map_location=self.device)
            optimizer_state_dict = torch.load(optimizer_ckpt, map_location=self.device)

            if scheduler_ckpt is not None:
                scheduler_state_dict = torch.load(
                    scheduler_ckpt, map_location=self.device
                )

        self._setup_model()
        self.model.load_state_dict(model_state_dict)
        print("Model state loaded!!")

        self._setup_training()

        self.optimizer.load_state_dict(optimizer_state_dict)
        print("Optimizer state loaded!!")

        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
            print("Scheduler state loaded!!")

        if total_iterations is None and use_cfg:
            total_iterations = (
                self.cfg.RESUME_TRAINING.NUM_STEPS
                if self.cfg.RESUME_TRAINING.NUM_STEPS is not None
                else self.cfg.RESUME_TRAINING.EPOCHS
            )

        if start_iteration is None and use_cfg:
            start_iteration = (
                self.cfg.RESUME_TRAINING.START_STEP
                if self.cfg.RESUME_TRAINING.START_STEP is not None
                else self.cfg.RESUME_TRAINING.START_EPOCH
            )

        return (total_iterations, start_iteration)

    def resume_training(
        self,
        consolidated_ckpt=None,
        model_ckpt=None,
        optimizer_ckpt=None,
        total_iterations=None,
        start_iteration=None,
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
        total_iterations : int, optional
            The number of epochs or steps to train for. Defaults to None (which uses the number of epochs specified in the config file)
        start_iteration : int, optional
            The epoch or step number to resume training from. Defaults to None (which starts from 0).
        scheduler_ckpt : str, optional
            The path to the scheduler checkpoint file. Defaults to None (which uses the scheduler checkpoint file specified in the config file).
        use_cfg : bool, optional
            Whether to use the config file or not. Defaults to False.
        """

        total_iterations, start_iteration = self._reload_trainer_states(
            consolidated_ckpt=consolidated_ckpt,
            model_ckpt=model_ckpt,
            optimizer_ckpt=optimizer_ckpt,
            total_iterations=total_iterations,
            start_iteration=start_iteration,
            scheduler_ckpt=scheduler_ckpt,
            use_cfg=use_cfg,
        )

        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)

        self._trainer(total_iterations, start_iteration)

        print("Training complete!")
        print(f"Total training time: {str(timedelta(seconds=sum(self.times)))}")


class Trainer(BaseTrainer):
    """
    Trainer class for training and evaluating models
    on a single device CPU/GPU.

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
        super(Trainer, self).__init__()

        self.cfg = cfg
        self.model_name = model.__class__.__name__.lower()
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _setup_device(self):

        if (
            isinstance(self.cfg.DEVICE, str) and self.cfg.DEVICE.lower() == "cpu"
        ) or int(self.cfg.DEVICE) == -1:
            self.device = torch.device("cpu")
            self.cfg.MIXED_PRECISION = False
            print("Running on CPU\n")

        elif not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.cfg.MIXED_PRECISION = False
            print("CUDA device(s) not available. Running on CPU\n")

        else:
            self.device = torch.device(int(self.cfg.DEVICE))
            torch.cuda.empty_cache()

    def _setup_model(self):
        self.model = self.model.to(self.device)

    def _is_main_process(self):
        return True

    def train(
        self,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        total_iterations=None,
        start_iteration=None,
    ):
        """
        Method to train the model using a single cpu/gpu device.

        Parameters
        ----------
        loss_fn : torch.nn.modules.loss, optional
            The loss function to be used. Defaults to None (which uses the loss function specified in the config file).
        optimizer : torch.optim.Optimizer, optional
            The optimizer to be used. Defaults to None (which uses the optimizer specified in the config file).
        scheduler : torch.optim.lr_scheduler, optional
            The learning rate scheduler to be used. Defaults to None (which uses the scheduler specified in the config file).
        total_iterations : int, optional
            The number of epochs or steps to train for. Defaults to None (which uses the number of epochs specified in the config file)
        start_iteration : int, optional
            The epoch or step number to resume training from. Defaults to None (which starts from 0).

        """
        self._setup_device()
        self._setup_model()
        self._setup_training(
            rank=0, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler
        )

        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)

        self._trainer(total_iterations, start_iteration)

        print("Training complete!")
        print(f"Total training time: {str(timedelta(seconds=sum(self.times)))}")


class DistributedTrainer(BaseTrainer):
    """
    Trainer class for distributed training and evaluating models
    on a single node multi-gpu environment.

    Parameters
    ----------
    cfg : CfgNode
        Configuration object for training
    model : torch.nn.Module
        Model to be trained
    train_loader_creator : ezflow.DataloaderCreator
        DataloaderCreator instance for training
    val_loader_creator : ezflow.DataloaderCreator
        DataloaderCreator instance for validation
    """

    def __init__(self, cfg, model, train_loader_creator, val_loader_creator):
        super(DistributedTrainer, self).__init__()
        self.model_parallel = True
        self.cfg = cfg
        self.model_name = model.__class__.__name__.lower()
        self.model = model

        self.local_rank = None
        self.device_ids = None

        self.train_loader = None
        self.val_loader = None

        self.train_loader_creator = train_loader_creator

        # Validate model only on the main process.
        val_loader_creator.distributed = False
        self.val_loader = val_loader_creator.get_dataloader()

        self._validate_ddp_config()

    def _validate_ddp_config(self):

        if self.cfg.DEVICE != "all":
            """
            Set CUDA_VISIBLE_DEVICES before performing any torch.cuda operations.
            """

            device = self.cfg.DEVICE
            if type(device) != str:
                device = str(device)

            os.environ["CUDA_VISIBLE_DEVICES"] = device

            device_ids = device.split(",")
            device_ids = [int(id) for id in device_ids]
            assert (
                len(device_ids) <= torch.cuda.device_count()
            ), "Total devices cannot be greater than available CUDA devices."
            self.device_ids = device_ids
            print(f"\nRunning on devices: {self.device_ids}\n")

        assert self.cfg.DISTRIBUTED.WORLD_SIZE <= torch.cuda.device_count(), (
            "WORLD_SIZE cannot be greater than available CUDA devices. "
            f"Given WORLD_SIZE: {self.cfg.DISTRIBUTED.WORLD_SIZE} "
            f"but total CUDA devices available: {torch.cuda.device_count()}"
        )

        if not is_port_available(int(self.cfg.DISTRIBUTED.MASTER_PORT)):

            print(
                f"\nPort: {self.cfg.DISTRIBUTED.MASTER_PORT} is not available to use!"
            )

            free_port = find_free_port()
            print(f"Assigning free port: {free_port}\n")

            self.cfg.DISTRIBUTED.MASTER_PORT = free_port

    def _setup_device(self, rank):
        assert (
            torch.cuda.is_available()
        ), "CUDA devices are not available. Use ezflow.Trainer for single device training."
        self.device = torch.device(rank)
        self.local_rank = rank
        torch.cuda.empty_cache()
        torch.cuda.set_device(rank)

    def _setup_ddp(self, rank):
        os.environ["MASTER_ADDR"] = self.cfg.DISTRIBUTED.MASTER_ADDR
        os.environ["MASTER_PORT"] = self.cfg.DISTRIBUTED.MASTER_PORT

        seed(0)

        dist.init_process_group(
            backend=self.cfg.DISTRIBUTED.BACKEND,
            init_method="env://",
            world_size=self.cfg.DISTRIBUTED.WORLD_SIZE,
            rank=rank,
        )
        print(f"{rank + 1}/{self.cfg.DISTRIBUTED.WORLD_SIZE} process initialized.")

        # synchronizes all the threads to reach this point before moving on
        dist.barrier()

    def _is_main_process(self):
        return self.local_rank == 0

    def _setup_model(self, rank):

        if self.cfg.DISTRIBUTED.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model = DDP(
            self.model.cuda(rank),
            device_ids=[rank],
        )

        self.model = self.model.to(self.device)

    def _cleanup(self):
        dist.destroy_process_group()

    def _main_worker(
        self,
        rank,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        total_iterations=None,
        start_iteration=None,
    ):
        self._setup_device(rank)
        self._setup_ddp(rank)
        self._setup_model(rank)
        self.train_loader = self.train_loader_creator.get_dataloader(rank=rank)

        self._setup_training(
            rank=rank, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler
        )

        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        # synchronizes all the threads to reach this point before moving on
        dist.barrier()
        self._trainer(total_iterations, start_iteration)

        if self._is_main_process():
            print("\nTraining complete!")
            print(f"Total training time: {str(timedelta(seconds=sum(self.times)))}")

        self._cleanup()

    def train(
        self,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        total_iterations=None,
        start_iteration=None,
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
        total_iterations : int, optional
            The number of epochs or steps to train for. Defaults to None (which uses the number of epochs specified in the config file)
        start_iteration : int, optional
            The epoch or step number to resume training from. Defaults to None (which starts from 0).

        """
        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)
        print("\nPerforming distributed training\n")
        print("-" * 80)

        mp.spawn(
            self._main_worker,
            args=(loss_fn, optimizer, scheduler, total_iterations, start_iteration),
            nprocs=self.cfg.DISTRIBUTED.WORLD_SIZE,
            join=True,
        )
