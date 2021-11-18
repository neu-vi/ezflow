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
    Builds a model from a model name and config.
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
