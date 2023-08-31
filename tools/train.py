import argparse

from ezflow.data import build_dataloader, get_dataset_list
from ezflow.engine import DistributedTrainer, Trainer, get_training_cfg
from ezflow.models import build_model, get_model_list


def main(args):

    # Load training configuration
    cfg = get_training_cfg(args.train_cfg)

    if args.device:
        cfg.DEVICE = args.device

    cfg.DATA.TRAIN_DATASET[args.train_ds].ROOT_DIR = args.train_data_dir
    cfg.DATA.VAL_DATASET[args.val_ds].ROOT_DIR = args.val_data_dir

    # Create dataloader
    train_loader = build_dataloader(
        cfg.DATA, distributed=cfg.DISTRIBUTED.USE, world_size=cfg.DISTRIBUTED.WORLD_SIZE
    )
    val_loader = build_dataloader(
        cfg.DATA, distributed=cfg.DISTRIBUTED.USE, world_size=cfg.DISTRIBUTED.WORLD_SIZE
    )

    # Build model
    model = build_model(args.model, default=True)

    # Create trainer
    if cfg.DISTRIBUTED.USE is True:
        trainer = DistributedTrainer(
            cfg,
            model,
            train_loader_creator=train_loader,
            val_loader_creator=val_loader,
        )
    else:
        trainer = Trainer(
            cfg,
            model,
            train_loader_creator=train_loader,
            val_loader_creator=val_loader,
        )

    # Train model
    trainer.train(total_epochs=args.n_epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train an optical flow model using EzFlow"
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--train_ds",
        type=str,
        required=True,
        choices=get_dataset_list(),
        help="Name of the training dataset.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to the root data directory",
    )
    parser.add_argument(
        "--val_ds",
        type=str,
        required=True,
        choices=get_dataset_list(),
        help="Name of the validation dataset.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Path to the root data directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_model_list(),
        help="Name of the model to train",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=None, help="Number of epochs to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device(s) to train on separated by commas. -1 for CPU",
    )

    args = parser.parse_args()

    main(args)
