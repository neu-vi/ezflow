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

    if cfg.DISTRIBUTED.USE is True:
        train_loader_creator = DataloaderCreator(
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            distributed=True,
            world_size=cfg.DISTRIBUTED.WORLD_SIZE,
            append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        )

        val_loader_creator = DataloaderCreator(
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            distributed=True,
            world_size=cfg.DISTRIBUTED.WORLD_SIZE,
            append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        )
    else:
        train_loader_creator = DataloaderCreator(
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        )

        val_loader_creator = DataloaderCreator(
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            append_valid_mask=cfg.DATA.APPEND_VALID_MASK,
        )

    # Add Training Dataset

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "flyingchairs":
        train_loader_creator.add_FlyingChairs(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "flyingthings3d":
        train_loader_creator.add_FlyingThings3D(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "sceneflow":
        train_loader_creator.add_SceneFlow(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "mpisintel":
        train_loader_creator.add_MPISintel(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "kitti":
        train_loader_creator.add_Kitti(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    if cfg.DATA.TRAIN_DATASET.NAME.lower() == "autoflow":
        train_loader_creator.add_AutoFlow(
            root_dir=cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            augment=cfg.DATA.AUGMENTATION.USE,
            aug_params=aug_params,
        )

    # Add Validation Dataset

    if cfg.DATA.VAL_DATASET.NAME.lower() == "flyingchairs":
        val_loader_creator.add_FlyingChairs(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if cfg.DATA.VAL_DATASET.NAME.lower() == "flyingthings3d":
        val_loader_creator.add_FlyingThings3D(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if cfg.DATA.VAL_DATASET.NAME.lower() == "sceneflow":
        val_loader_creator.add_SceneFlow(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if cfg.DATA.VAL_DATASET.NAME.lower() == "mpisintel":
        val_loader_creator.add_MPISintel(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="training",
            dstype="clean",
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if cfg.DATA.VAL_DATASET.NAME.lower() == "kitti":
        val_loader_creator.add_Kitti(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if cfg.DATA.VAL_DATASET.NAME.lower() == "autoflow":
        val_loader_creator.add_AutoFlow(
            root_dir=cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    # Build model

    model = build_model(args.model, default=True)

    # Create trainer

    trainer = Trainer(cfg, model, train_loader_creator, val_loader_creator)

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
