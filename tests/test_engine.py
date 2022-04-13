from torch import nn
from torch.utils.data import DataLoader

from ezflow.engine import (
    Trainer,
    eval_model,
    get_training_cfg,
    prune_l1_structured,
    prune_l1_unstructured,
)

from .utils import MockOpticalFlowDataset, MockOpticalFlowModel

img_size = (64, 64)
img_channels = 3
len_dataset = 4
batch_size = 2

mock_dataloader = DataLoader(
    MockOpticalFlowDataset(size=img_size, channels=img_channels, length=len_dataset),
    batch_size=batch_size,
)
mock_model = MockOpticalFlowModel(img_channels=img_channels)


def test_epoch_trainer():

    training_cfg = get_training_cfg(
        cfg_path="./tests/configs/base_trainer_test.yaml", custom=True
    )
    train_loader = val_loader = mock_dataloader

    trainer = Trainer(training_cfg, mock_model, train_loader, val_loader)
    trainer.train()
    trainer.resume_training(
        consolidated_ckpt=f"./ckpts/{mock_model.__class__.__name__.lower()}_epoch0.pth",
        total_iterations=1,
    )


def test_step_trainer():

    training_cfg = get_training_cfg(
        cfg_path="./tests/configs/step_trainer_test.yaml", custom=True
    )
    train_loader = val_loader = mock_dataloader

    trainer = Trainer(training_cfg, mock_model, train_loader, val_loader)
    trainer.train(total_iterations=1)


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
