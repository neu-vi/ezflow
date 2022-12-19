from ezflow.data import DataloaderCreator
from ezflow.engine import DistributedTrainer, Trainer, get_cfg
from ezflow.models import build_model


def main(args):

    # Load training configuration

    cfg = get_cfg(args.train_cfg)

    if args.device:
        cfg.DEVICE = args.device

    cfg.DATA.TRAIN_DATASET.ROOT_DIR = args.train_data_dir
    cfg.DATA.VAL_DATASET.ROOT_DIR = args.val_data_dir

    if args.n_steps is not None:
        cfg.NUM_STEPS = args.n_steps

        if cfg.SCHEDULER.NAME == "OneCycleLR":
            cfg.SCHEDULER.PARAMS.total_steps = cfg.NUM_STEPS

    # Create dataloaders

    train_aug_params = None
    val_aug_params = None
    if cfg.DATA.AUGMENTATION.USE and cfg.DATA.AUGMENTATION.PARAMS:
        train_aug_params = cfg.DATA.AUGMENTATION.PARAMS.TRAINING.to_dict()
        val_aug_params = cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.to_dict()

    train_loader_creator = DataloaderCreator(
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        distributed=cfg.DISTRIBUTED.USE,
        world_size=cfg.DISTRIBUTED.WORLD_SIZE,
        append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        shuffle=cfg.DATA.SHUFFLE,
    )

    val_loader_creator = DataloaderCreator(
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        distributed=cfg.DISTRIBUTED.USE,
        world_size=cfg.DISTRIBUTED.WORLD_SIZE,
        append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        shuffle=cfg.DATA.SHUFFLE,
    )

    # TODO: Create a Dataloader Registry
    train_loader_creator.add_FlyingChairs(
        root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
        crop=True,
        crop_type="random",
        crop_size=cfg.DATA.TRAIN_CROP_SIZE,
        augment=cfg.DATA.AUGMENTATION.USE,
        aug_params=train_aug_params,
        norm_params=cfg.DATA.NORM_PARAMS,
    )

    val_loader_creator.add_FlyingChairs(
        val_loader_creator.add_FlyingChairs(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=val_aug_params,
            norm_params=cfg.DATA.NORM_PARAMS,
        )
    )

    # Build model

    model = build_model(args.model, default=True)

    # Create trainer
    if cfg.DISTRIBUTED.USE is True:
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
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to the root data directory",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Path to the root data directory",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--n_steps", type=int, default=None, help="Number of iterations to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device(s) to train on separated by commas. -1 for CPU",
    )

    args = parser.parse_args()

    main(args)
