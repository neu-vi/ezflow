from ..model_zoo import get_cfg
from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(
    model_name, model_cfg_path, model_cfg=None
):  # To-do: Add load model weight

    """
    Builds a model from a model name and config.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")

    if model_cfg is None:
        model_cfg = get_cfg(model_cfg_path)

    model = MODEL_REGISTRY.get(model_name)

    return model(model_cfg)
