"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

from ..config import get_cfg


class _TrainerConfigs:
    """
    Container class for training configurations.
    """

    NAME_TO_TRAINER_CONFIG = {
        "BASE": "base.yaml",
        "RAFT": "raft_default.yaml",
        "DICL": "dicl_default.yaml",
    }

    @staticmethod
    def query(trainer_name):

        trainer_name = trainer_name.upper()

        if trainer_name in _TrainerConfigs.NAME_TO_TRAINER_CONFIG:

            cfg_file = _TrainerConfigs.NAME_TO_TRAINER_CONFIG[trainer_name]
            return cfg_file

        raise ValueError(
            f"Trainer name '{trainer_name}' not found in the training configs"
        )


def get_training_cfg(cfg_path=None, cfg_name=None, custom=True):

    """
    Parameters
    ----------
    cfg_path : str
        Path to the config file.
    cfg_name : str
        Name of the config file.
    custom : bool
        If True, the config file is assumed to be a custom config file.
        If False, the config file is assumed to be a standard config file present in ezflow/configs/trainers.

    Returns
    -------
    cfg : CfgNode
        The config object
    """

    assert (
        cfg_path is not None or cfg_name is not None
    ), "Either cfg_path or cfg_name must be provided"

    if cfg_path is None:
        cfg_path = _TrainerConfigs.query(cfg_name)
        return get_cfg(cfg_path, custom=False, grp="trainers")

    return get_cfg(cfg_path, custom=custom, grp="trainers")
