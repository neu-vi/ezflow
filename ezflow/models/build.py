import torch

from ..config import get_cfg
from ..model_zoo import _ModelZooConfigs
from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")


def get_default_model_cfg(model_name):

    cfg_path = _ModelZooConfigs.query(model_name)

    return get_cfg(cfg_path)


def build_model(
    name, cfg_path=None, custom_cfg=False, cfg=None, default=False, weights_path=None
):
    """
    Builds a model from a model name and config. Also supports loading weights

    Parameters
    ----------
    name : str
        Name of the model to build
    cfg_path : str, optional
        Path to a config file. If not provided, will use the default config
        for the model
    custom_cfg : bool, optional
        Whether to use a custom config file. If False, will use the default
        config for the model
    cfg : CfgNode object, optional
        Custom config object. If provided, will use this config instead of
        the default config for the model
    default : bool, optional
        Whether to use the default config for the model
    weights_path : str, optional
        Path to a weights file

    Returns
    -------
    torch.nn.Module
        The model
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry.")

    if cfg is None:

        if default:
            cfg_path = _ModelZooConfigs.query(name)
            cfg = get_cfg(cfg_path)

        else:
            assert cfg_path is not None, "Please provide a config path."
            cfg = get_cfg(cfg_path, custom=custom_cfg)

    model = MODEL_REGISTRY.get(name)
    model = model(cfg)

    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )

    return model
