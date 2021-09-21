"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

import os

import pkg_resources


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

    cfg = CfgNode  # should be a empty CfgNode or a default
    cfg.merge_from_file(cfg_file)

    return cfg


def get(cfg_path, trained: bool = False, device: Optional[str] = None):
    """
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.

    Returns:
        nn.Module: a detectron2 model. Will be in training mode.

    Example:
    ::
        from detectron2 import model_zoo
        model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
    """
    cfg = get_cfg(cfg_path, trained)

    if device is None and not torch.cuda.is_available():
        device = "cpu"
    if device is not None and isinstance(cfg, CfgNode):
        cfg.MODEL.DEVICE = device

    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        model = instantiate(cfg.model)
        if device is not None:
            model = model.to(device)
        if "train" in cfg and "init_checkpoint" in cfg.train:
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return model
