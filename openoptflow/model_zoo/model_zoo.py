"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

import os

import pkg_resources

from ..config import get_default_cfg


class _ModelZooConfigs:

    MODEL_NAME_TO_CONFIG = {
        "RAFT": "raft.yaml",
        "RAFT_SMALL": "raft_small.yaml",
    }

    @staticmethod
    def query(model_name):

        if model_name in _ModelZooConfigs.MODEL_NAME_TO_CONFIG:

            cfg_file = _ModelZooConfigs.MODEL_NAME_TO_CONFIG[model_name]
            return cfg_file

        raise ValueError(f"Model name '{model_name}' not found in model zoo")


def get_cfg_path(cfg_path, grp="models"):

    """
    Returns the complete path to a config file present in openoptflow

    Parameters
    ----------
    cf_path : str
        Config file name relative to openoptflow's "configs/{grp}" directory

    Returns
    -------
    str
        The complete path to the config file.
    """

    cfg_complete_path = pkg_resources.resource_filename(
        "openoptflow.model_zoo", os.path.join("configs", grp, cfg_path)
    )

    if not os.path.exists(cfg_complete_path):
        raise RuntimeError(
            f"{grp}/{cfg_path} is not available in openoptflow's model zoo!"
        )

    return cfg_complete_path


def get_cfg(cfg_path, custom=False):

    """
    Returns a config object for a model in model zoo.

    Args
    ----
    config_path : str
        Config file name relative to openoptflow's "configs/" directory


    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """

    if not custom:
        cfg_path = get_cfg_path(cfg_path)

    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    return cfg
