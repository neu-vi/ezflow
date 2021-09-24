import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ..data.dataloader import DeviceDataLoader
from ..utils import AverageMeter


class BaseTrainer:
    """
    Base Trainer class
    """

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
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_dir = log_dir

        if device == "-1" or device == "cpu":
            device = torch.device("cpu")
            print("Running on CPU")

        elif not torch.cuda.is_available():
            device = torch.device("cpu")
            print("CUDA device(s) not available. Running on CPU")

        else:
            if device == "all":
                device = torch.device("cuda")
                model = nn.DataParallel(model)

            else:
                device_ids = device.split(",")
                device_ids = [int(id) for id in device_ids]
                cuda_str = "cuda:" + device
                device = torch.device(cuda_str)
                model = nn.DataParallel(model, device_ids=device_ids)
                print(f"Running on CUDA devices {device_ids}")

        self.device = device
        self.train_loader = DeviceDataLoader(self.train_loader, self.device)
        self.val_loader = DeviceDataLoader(self.val_loader, self.device)

    def _calculate_loss(self, pred, label):

        raise NotImplementedError

    def _calculate_metric(self, pred, label):

        if isinstance(pred, tuple):
            pred = pred[0]

        metric = torch.sqrt(torch.sum((pred - label) ** 2, dim=1))

        return metric

    def _train_model(self, n_epochs):

        writer = SummaryWriter(log_dir=self.log_dir)

        best_model = deepcopy(self.model)

        model = self.model.to(self.device)
        model.train()

        iter_loss = AverageMeter()
        avg_metric = 0.0

        epochs = 0
        while epochs < n_epochs:

            print(f"Epoch {epochs+1} of {n_epochs} ")

            iter_loss.reset()
            for iteration, (img, label) in enumerate(self.train_loader):

                pred = model(img)

                loss = self._calculate_loss(pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter_loss.update(loss.item(), self.train_loader.batch_size)

                if iteration % 5000 == 0:
                    writer.add_scalar(
                        "avg_training_loss",
                        iter_loss.avg,
                        iteration + (epochs * len(self.train_loader.dataset)),
                    )

            if epochs % self.eval_interval == 0:

                new_avg_metric = self._eval_model(model)
                if new_avg_metric > avg_metric:
                    best_model = deepcopy(model)
                    avg_metric = new_avg_metric

                writer.add_scalar("validation_metric", avg_metric, epochs + 1)

            print(f"Loss = {iter_loss.sum}")
            writer.add_scalar("epochs_training_loss", iter_loss.sum, epochs + 1)

            if epochs % self.save_interval == 0:
                model_name = model.__class__.__name__.lower()
                torch.save(
                    model.state_dict(),
                    os.path.join(self.save_dir, model_name + ".pth"),
                )

            epochs += 1

        writer.close()

        return best_model

    def _eval_model(self, model):

        model = model.to(self.device)
        metric_meter = AverageMeter()
        batch_size = self.val_loader.batch_size

        model.eval()
        with torch.no_grad():
            for img, label in self.val_loader:

                pred = model(img)

                metric = self._calculate_metric(pred, label)
                metric_meter.update(metric.item(), n=batch_size)

        return metric_meter.avg

    def train(self, n_epochs=10):

        model_name = self.model.__class__.__name__.lower()

        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Training {model_name} for {n_epochs}")
        model = self._train_model(n_epochs)
        print("Training complete!")

        torch.save(
            model.state_dict(), os.path.join(self.save_dir, model_name + "_best.pth")
        )
        print("Saved best model!")
