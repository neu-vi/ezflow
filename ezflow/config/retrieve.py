"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

import os

import pkg_resources

from .config import CfgNode


def get_cfg_obj():
    """Returns a :class:`CfgNode` object"""
    return CfgNode(new_allowed=True)


def get_cfg_path(cfg_path, grp="models"):

    """
    Returns the complete path to a config file present in ezflow

    Parameters
    ----------
    cfg_path : str
        Config file path relative to ezflow's "configs/{grp}" directory

    Returns
    -------
    str
        The complete path to the config file.
    """

    grp = grp.lower()
    assert grp in ("models", "trainers"), "Grp must be either 'models' or 'trainers' "

    if grp == "models":
        cfg_complete_path = pkg_resources.resource_filename(
            "ezflow.model_zoo", os.path.join("configs", grp, cfg_path)
        )

    elif grp == "trainers":
        cfg_complete_path = pkg_resources.resource_filename(
            "ezflow.model_zoo", os.path.join("configs", grp, cfg_path)
        )

    if not os.path.exists(cfg_complete_path):
        raise RuntimeError(
            f"{grp}/{cfg_path} is not available in ezflow's model zoo or trainer configs!"
        )

    return cfg_complete_path


def get_cfg(cfg_path, custom=False, grp="models"):

    """
    Returns a config object for a model in model zoo.

    Parameters
    ----
    config_path : str
        Complete config file path


    Returns
        CfgNode or omegaconf.DictConfig: a config object
    """

    if not custom:
        cfg_path = get_cfg_path(cfg_path, grp=grp)

    cfg = get_cfg_obj()
    cfg.merge_from_file(cfg_path)

    return cfg
