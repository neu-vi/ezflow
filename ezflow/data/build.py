from ..utils import Registry

DATASET_REGISTRY = Registry("DATASET_REGISTRY")


def build_dataloader(cfg, split="training", is_distributed=False, world_size=None):
    """
    Build a dataloader from registered datasets

    Parameters
    ----------
    cfg: :class:`CfgNode`
        Dataset configuration to instantiate DatalaoderCreator


    Returns
    -------
    ezflow.data.DataloaderCreator

    """
    from .dataloader import DataloaderCreator

    # TODO: assert mandatory config in cfg.data

    dataloader_creator = DataloaderCreator(
        batch_size=cfg.BATCH_SIZE,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
        drop_last=cfg.DROP_LAST,
        init_seed=cfg.INIT_SEED,
        append_valid_mask=cfg.APPEND_VALID_MASK,
        distributed=is_distributed,
        world_size=world_size,
    )

    data_cfg = cfg.TRAIN_DATASET if split == "training" else cfg.VAL_DATASET

    for key in data_cfg:
        data_cfg[key].SPLIT = split
        data_cfg[key].INIT_SEED = cfg.INIT_SEED
        data_cfg[key].NORM_PARAMS = cfg.NORM_PARAMS

        dataset = DATASET_REGISTRY.get(key)(data_cfg[key])
        dataloader_creator.add_dataset(dataset)

    return dataloader_creator


def get_dataset_list():
    return DATASET_REGISTRY.get_list()