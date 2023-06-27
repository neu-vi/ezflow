from ..utils import Registry
from .dataloader import DataloaderCreator

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
    data_cfg.INIT_SEED = cfg.INIT_SEED
    data_cfg.NORM_PARAMS = cfg.NORM_PARAMS

    dataset = DATASET_REGISTRY.get(data_cfg.NAME)
    dataset = dataset(data_cfg)
    dataloader_creator.add_dataset(dataset)

    return dataloader_creator
