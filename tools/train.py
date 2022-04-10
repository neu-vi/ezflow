from ezflow.data import DataloaderCreator
from ezflow.engine import DistributedTrainer, Trainer, get_training_cfg
from ezflow.models import build_model


def main(args):

    # Load training configuration

    cfg = get_training_cfg(args.train_cfg)

    if args.device:
        cfg.DEVICE = args.device

    # Create dataloaders

    aug_params = None
    if cfg.DATA.AUGMENTATION.USE and cfg.DATA.AUGMENTATION.PARAMS:
        aug_params = cfg.DATA.AUGMENTATION.PARAMS.to_dict()

    train_loader_creator = DataloaderCreator(
        cfg.DATA.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )
    train_loader_creator.add_FlyingChairs(
        root_dir=args.data_dir, augment=cfg.DATA.AUGMENTATION.USE, aug_params=aug_params
    )

    val_loader_creator = DataloaderCreator(
        cfg.DATA.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )
    val_loader_creator.add_FlyingChairs(
        root_dir=args.data_dir,
        split="validation",
        augment=cfg.DATA.AUGMENTATION.USE,
        aug_params=aug_params,
    )

    # Build model

    model = build_model(args.model, default=True)

    # Create trainer

    if training_cfg.DISTRIBUTED.USE is True:
        trainer = DistributedTrainer(
            cfg,
            model,
            train_loader_creator=train_loader_creator,
            val_loader_creator=val_loader_creator,
        )
    else:
        trainer = Trainer(
            cfg,
            model,
            train_loader=train_loader_creator.get_dataloader(),
            val_loader=val_loader_creator.get_dataloader(),
        )

    # Train model
    trainer.train()


if __name__ == "__main__":

    import argparse

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
        "--data_dir", type=str, required=True, help="Path to the root data directory"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=None, help="Number of epochs to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device(s) to train on separated by commas. -1 for CPU",
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Whether to do distributed training",
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        help="Backend to use for distributed computing",
    )

    args = parser.parse_args()

    main(args)
