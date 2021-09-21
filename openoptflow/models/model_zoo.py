"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

import os

import pkg_resources

from ..config import get_default_cfg


def get_cfg_file(cfg_path):

    """
    Returns the complete path to a config file present in openoptflow

    Parameters
    ----------
    cf_path : str
        Config file name relative to openoptflow's "configs/" directory

    Returns
    -------
    str
        The complete path to the config file.
    """

    cfg_complete_path = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", cfg_path)
    )
    if not os.path.exists(cfg_complete_path):
        raise RuntimeError(f"{cfg_path} is not available in openoptflow's model zoo!")

    return cfg_complete_path


def get_cfg(cfg_path):

    """
    Returns a config object for a model in model zoo.

    Args
    ----
    config_path : str
        Config file name relative to openoptflow's "configs/" directory


    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """

    cfg_file = get_cfg_file(cfg_path)

    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_file)

    return cfg
