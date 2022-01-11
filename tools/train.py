from ezflow.data import DataloaderCreator
from ezflow.engine import Trainer, get_training_cfg
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

    train_loader = DataloaderCreator(cfg.DATA.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
    train_loader.add_flying_chairs(
        root_dir=args.data_dir, augment=cfg.DATA.AUGMENTATION.USE, aug_params=aug_params
    )
    train_loader = train_loader.get_dataloader()

    val_loader = DataloaderCreator(cfg.DATA.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
    val_loader.add_flying_chairs(
        root_dir=args.data_dir,
        split="validation",
        augment=cfg.DATA.AUGMENTATION.USE,
        aug_params=aug_params,
    )
    val_loader = val_loader.get_dataloader()

    # Build model

    model = build_model(args.model, default=True)

    # Create trainer

    trainer = Trainer(cfg, model, train_loader, val_loader)

    # Train model

    n_epochs = None
    if args.n_epochs:
        n_epochs = args.n_epochs

    trainer.train(n_epochs=n_epochs)


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
