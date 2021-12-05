from torch.nn.utils import prune


def prune_l1_unstructured(model, layer_type, proportion):
    """
    L1 unstructured pruning

    Parameters
    ----------
    model : torch.nn.Module
        The model to prune
    layer_type : torch.nn.Module
        The layer type to prune
    proportion : float
        The proportion of weights to prune
    """

    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, "weight", proportion)
            prune.remove(module, "weight")

    return model


def prune_l1_structured(model, layer_type, proportion):
    """
    L1 structured pruning

    Parameters
    ----------
    model : torch.nn.Module
        The model to prune
    layer_type : torch.nn.Module
        The layer type to prune
    proportion : float
        The proportion of weights to prune
    """

    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, "weight", proportion, n=1, dim=1)
            prune.remove(module, "weight")

    return model
