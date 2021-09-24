from ..model_zoo import _ModelZooConfigs, get_cfg
from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(
    name, cfg_path=None, custom_cfg=False, cfg=None, default=False
):  # To-do: Add load model weight

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

    return model(cfg)
