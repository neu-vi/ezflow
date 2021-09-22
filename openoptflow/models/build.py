from ..model_zoo import get_cfg
from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(
    name, cfg_path, custom_cfg=False, cfg=None
):  # To-do: Add load model weight

    """
    Builds a model from a model name and config.
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry.")

    if cfg is None:
        cfg = get_cfg(cfg_path, custom=custom_cfg)

    model = MODEL_REGISTRY.get(name)

    return model(cfg)
