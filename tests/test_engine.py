from unittest import TestCase, mock

import torch
from torch import nn
from torch.utils.data import DataLoader

import ezflow
from ezflow.engine import (
    DistributedTrainer,
    Trainer,
    eval_model,
    get_training_cfg,
    prune_l1_structured,
    prune_l1_unstructured,
)
from ezflow.functional import MultiScaleLoss

from .utils import MockDataloaderCreator, MockOpticalFlowDataset, MockOpticalFlowModel

img_size = (64, 64)
img_channels = 3
len_dataset = 4
batch_size = 2

mock_dataloader = DataLoader(
    MockOpticalFlowDataset(size=img_size, channels=img_channels, length=len_dataset),
    batch_size=batch_size,
)
mock_model = MockOpticalFlowModel(img_channels=img_channels)


class TestTrainer(TestCase):
    def setUp(self):
        self.training_cfg = get_training_cfg(
            cfg_path="./tests/configs/base_trainer_test.yaml", custom=True
        )
        self.train_loader = mock_dataloader
        self.val_loader = mock_dataloader
        self.mock_model = mock_model

    @mock.patch.object(Trainer, "_setup_model")
    @mock.patch.object(Trainer, "_setup_training")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch.object(torch, "device", return_value=torch.device)
    @mock.patch.object(torch.cuda, "is_available", return_value=False)
    @mock.patch("ezflow.engine.trainer.os")
    def test_valid_device_cpu(
        self,
        mock_os,
        mock_cuda_available,
        mock_torch_device,
        mock_trainer,
        mock_setup_training,
        mock_setup_model,
    ):

        trainer = Trainer(
            self.training_cfg, self.mock_model, self.train_loader, self.val_loader
        )
        trainer._trainer = Trainer._epoch_trainer
        trainer.train(total_iterations=50, start_iteration=0)

        # assert torch
        mock_torch_device.assert_called_with("cpu")

        # assert trainer
        mock_setup_model.assert_called()
        mock_setup_training.assert_called_with(
            rank=0, loss_fn=None, optimizer=None, scheduler=None
        )
        mock_trainer.assert_called_with(50, 0)
        assert trainer.cfg.MIXED_PRECISION == False

        # assert system calls
        mock_os.makedirs.assert_called_with("./logs", exist_ok=True)

        del trainer

    @mock.patch.object(Trainer, "_setup_model")
    @mock.patch.object(Trainer, "_setup_training")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch.object(torch, "device", return_value=torch.device)
    @mock.patch.object(torch.cuda, "is_available", return_value=False)
    @mock.patch("ezflow.engine.trainer.os")
    def test_invalid_device_cpu_fallback(
        self,
        mock_os,
        mock_cuda_available,
        mock_torch_device,
        mock_trainer,
        mock_setup_training,
        mock_setup_model,
    ):

        self.training_cfg.DEVICE = 0
        trainer = Trainer(
            self.training_cfg, self.mock_model, self.train_loader, self.val_loader
        )
        trainer._trainer = Trainer._epoch_trainer
        trainer.train()

        # assert torch
        mock_cuda_available.assert_called()
        mock_torch_device.assert_called_with("cpu")

        del trainer

    @mock.patch.object(Trainer, "_setup_model")
    @mock.patch.object(Trainer, "_setup_training")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch.object(torch, "device", return_value=torch.device)
    @mock.patch.object(torch.cuda, "is_available", return_value=True)
    @mock.patch.object(torch.cuda, "empty_cache")
    @mock.patch("ezflow.engine.trainer.os")
    def test_valid_device_cuda(
        self,
        mock_os,
        mock_empty_cache,
        mock_cuda_available,
        mock_torch_device,
        mock_trainer,
        mock_setup_training,
        mock_setup_model,
    ):
        self.training_cfg.DEVICE = 0
        trainer = Trainer(
            self.training_cfg, self.mock_model, self.train_loader, self.val_loader
        )
        trainer._trainer = Trainer._epoch_trainer
        trainer.train()

        # assert torch
        mock_cuda_available.assert_called()
        mock_torch_device.assert_called_with(self.training_cfg.DEVICE)
        mock_empty_cache.assert_called()

        del trainer

    @mock.patch.object(Trainer, "_setup_device")
    @mock.patch.object(Trainer, "_setup_training")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch.object(nn.Module, "to")
    @mock.patch("ezflow.engine.trainer.os")
    def test_setup_model(
        self,
        mock_os,
        mock_model_to_device,
        mock_trainer,
        mock_setup_training,
        mock_setup_device,
    ):
        trainer = Trainer(
            self.training_cfg, self.mock_model, self.train_loader, self.val_loader
        )
        trainer._trainer = Trainer._epoch_trainer
        trainer.device = torch.device("cpu")
        trainer.train()

        # assert torch
        mock_model_to_device.assert_called_with(torch.device("cpu"))

        # assert trainer
        mock_setup_device.assert_called()
        mock_setup_training.assert_called_with(
            rank=0, loss_fn=None, optimizer=None, scheduler=None
        )
        mock_trainer.assert_called_with(None, None)

        del trainer

    @mock.patch.object(Trainer, "_setup_device")
    @mock.patch.object(Trainer, "_setup_model")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch("ezflow.engine.trainer.SummaryWriter")
    @mock.patch("ezflow.engine.trainer.os")
    def test_setup_training(
        self, mock_os, mock_writer, mock_trainer, mock_setup_model, mock_setup_device
    ):
        trainer = Trainer(
            self.training_cfg, self.mock_model, self.train_loader, self.val_loader
        )
        trainer._trainer = Trainer._epoch_trainer
        trainer.train()

        # assert trainer
        assert isinstance(trainer.loss_fn, torch.nn.L1Loss)

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.state_dict()["param_groups"][0]["lr"] == 0.0003
        assert trainer.optimizer.state_dict()["param_groups"][0]["weight_decay"] == 0
        assert trainer.optimizer.state_dict()["param_groups"][0]["betas"] == [
            0.9,
            0.999,
        ]
        assert trainer.optimizer.state_dict()["param_groups"][0]["eps"] == 1.0e-08

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
        assert trainer.scheduler.state_dict()["base_lrs"] == [0.0003]
        assert trainer.scheduler.state_dict()["step_size"] == 10
        assert trainer.scheduler.state_dict()["gamma"] == 0.1

        assert isinstance(trainer.scaler, torch.cuda.amp.grad_scaler.GradScaler)

        mock_writer.assert_called_with(log_dir="./logs")

        del trainer

    @mock.patch.object(Trainer, "_setup_device")
    @mock.patch.object(Trainer, "_setup_model")
    @mock.patch.object(Trainer, "_epoch_trainer")
    @mock.patch("ezflow.engine.trainer.SummaryWriter")
    @mock.patch("ezflow.engine.trainer.os")
    def test_setup_training_with_custom_loss_fn(
        self, mock_os, mock_writer, mock_trainer, mock_setup_model, mock_setup_device
    ):
        cfg = get_training_cfg(
            cfg_path="./tests/configs/custom_loss_trainer.yaml", custom=True
        )

        trainer = Trainer(cfg, self.mock_model, self.train_loader, self.val_loader)
        trainer.train()

        assert isinstance(trainer.loss_fn, MultiScaleLoss)

        del trainer

    @mock.patch.object(torch, "save")
    @mock.patch("ezflow.engine.trainer.SummaryWriter")
    @mock.patch("ezflow.engine.trainer.os")
    def test_epoch_trainer(self, mock_os, mock_writer, mock_save_model):
        cfg = get_training_cfg(
            cfg_path="./tests/configs/custom_loss_trainer.yaml", custom=True
        )

        trainer = Trainer(cfg, self.mock_model, self.train_loader, self.val_loader)
        trainer._trainer = Trainer._epoch_trainer

        trainer.train()

        # assert save best model validate on metric
        trainer.writer.add_scalar.assert_called()
        torch.save.assert_called()

        del trainer

        cfg.VALIDATE_ON = "loss"
        trainer = Trainer(cfg, self.mock_model, self.train_loader, self.val_loader)
        trainer._trainer = Trainer._epoch_trainer

        trainer.train()

        # assert save best model validate on metric
        trainer.writer.add_scalar.assert_called()
        torch.save.assert_called()

        del trainer

    @mock.patch.object(torch, "save")
    @mock.patch("ezflow.engine.trainer.SummaryWriter")
    @mock.patch("ezflow.engine.trainer.os")
    def test_step_trainer(self, mock_os, mock_writer, mock_save_model):
        cfg = get_training_cfg(
            cfg_path="./tests/configs/custom_loss_trainer.yaml", custom=True
        )
        cfg.NUM_STEPS = 1

        trainer = Trainer(cfg, self.mock_model, self.train_loader, self.val_loader)
        trainer._trainer = Trainer._step_trainer

        trainer.train()

        # assert save best model using step iteration
        trainer.writer.add_scalar.assert_called()
        torch.save.assert_called()

        del trainer


class TestDistributedTrainer(TestCase):
    def setUp(self):
        self.training_cfg = get_training_cfg(
            cfg_path="./tests/configs/base_trainer_test.yaml", custom=True
        )

        self.training_cfg.DEVICE = "0,1"
        self.training_cfg.DISTRIBUTED.USE = True
        self.training_cfg.WORLD_SIZE = 2
        self.training_cfg.SYNC_BATCH_NORM = True

        dataloader_creator = MockDataloaderCreator()

        self.train_loader_creator = dataloader_creator
        self.val_loader_creator = dataloader_creator
        self.mock_model = mock_model

    @mock.patch.object(torch.cuda, "device_count", return_value=2)
    @mock.patch("ezflow.utils.other_utils")
    @mock.patch("ezflow.engine.trainer.os")
    def test_validate_ddp_config(self, mock_os, mock_utils, mock_cuda_device_count):

        mock_utils.is_port_available.return_value = False
        mock_utils.find_free_port.return_value = 55555

        trainer = DistributedTrainer(
            self.training_cfg,
            self.mock_model,
            self.train_loader_creator,
            self.val_loader_creator,
        )

        assert len(trainer.device_ids) == 2

        del trainer

    @mock.patch.object(torch.distributed, "barrier")
    @mock.patch.object(torch.distributed, "init_process_group")
    @mock.patch.object(torch, "device", return_value=torch.device)
    @mock.patch.object(torch.cuda, "is_available", return_value=True)
    @mock.patch.object(torch.cuda, "set_device")
    @mock.patch.object(torch.cuda, "empty_cache")
    @mock.patch.object(torch.cuda, "device_count", return_value=2)
    @mock.patch("ezflow.engine.trainer.os")
    def test_setup_ddp(
        self,
        mock_os,
        mock_device_count,
        mock_empty_cache,
        mock_torch_cuda_set_device,
        mock_cuda_available,
        mock_torch_device,
        mock_init_process_group,
        mock_dist_barrier,
    ):
        trainer = DistributedTrainer(
            self.training_cfg,
            self.mock_model,
            self.train_loader_creator,
            self.val_loader_creator,
        )

        trainer._setup_device(rank=0)
        mock_torch_cuda_set_device.assert_called_with(0)
        mock_torch_device.assert_called_with(0)
        assert trainer.local_rank == 0

        trainer._setup_ddp(rank=0)
        mock_init_process_group.assert_called_with(
            backend=self.training_cfg.DISTRIBUTED.BACKEND,
            init_method="env://",
            world_size=self.training_cfg.DISTRIBUTED.WORLD_SIZE,
            rank=0,
        )

        del trainer

    @mock.patch.object(torch.distributed, "destroy_process_group")
    @mock.patch.object(torch.distributed, "barrier")
    @mock.patch.object(DistributedTrainer, "_setup_model")
    @mock.patch.object(DistributedTrainer, "_setup_training")
    @mock.patch.object(DistributedTrainer, "_epoch_trainer")
    @mock.patch.object(DistributedTrainer, "_setup_ddp")
    @mock.patch.object(DistributedTrainer, "_setup_device")
    @mock.patch.object(torch.cuda, "device_count", return_value=2)
    @mock.patch("ezflow.engine.trainer.os")
    def test_main_worker(
        self,
        mock_os,
        mock_dev_count,
        mock_setup_device,
        mock_setup_ddp,
        mock_trainer,
        mock_setup_training,
        mock_setup_model,
        mock_dist_barrier,
        mock_dist_cleanup,
    ):
        trainer = DistributedTrainer(
            self.training_cfg,
            self.mock_model,
            self.train_loader_creator,
            self.val_loader_creator,
        )
        trainer._trainer = DistributedTrainer._epoch_trainer

        trainer._main_worker(rank=0)

        rank = 0
        mock_setup_device.assert_called_with(rank)
        mock_setup_ddp.assert_called_with(rank)
        mock_setup_model.assert_called_with(rank)
        mock_setup_training.assert_called_with(
            rank=0, loss_fn=None, optimizer=None, scheduler=None
        )
        mock_trainer.assert_called_with(None, None)
        mock_dist_barrier.assert_called()
        mock_dist_cleanup.assert_called()

        del trainer

    @mock.patch.object(torch.multiprocessing, "spawn")
    @mock.patch.object(torch.cuda, "device_count", return_value=2)
    @mock.patch("ezflow.engine.trainer.os")
    def test_multiprocessing(self, mock_os, mock_dev_count, mock_mp_spawn):
        trainer = DistributedTrainer(
            self.training_cfg,
            self.mock_model,
            self.train_loader_creator,
            self.val_loader_creator,
        )
        trainer.train()

        mock_mp_spawn.assert_called_with(
            trainer._main_worker,
            args=(None, None, None, None, None),
            nprocs=self.training_cfg.DISTRIBUTED.WORLD_SIZE,
            join=True,
        )


def test_eval_model():

    _ = eval_model(mock_model, mock_dataloader, device="cpu")


def test_l1_pruning():

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )
    _ = prune_l1_structured(model, nn.Linear, 0.5)
    _ = prune_l1_unstructured(model, nn.Linear, 0.5)
