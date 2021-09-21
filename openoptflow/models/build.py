from ..model_zoo import get_cfg
from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(model_name, model_cfg_path):  # To-do: Add load model weight

    """
    Builds a model from a model name and config.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")

    model_cfg = get_cfg(model_cfg_path)

    return MODEL_REGISTRY.get(model_cfg)
