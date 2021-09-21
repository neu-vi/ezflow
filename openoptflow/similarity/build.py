from ..utils import Registry

SIMILARITY_REGISTRY = Registry("SIMILARITY")


def build_similarity(cfg, name=None):

    """
    Build a similarity function from a registered similarity function name.

    Parameters
    ----------
    name : str
        Name of the registered similarity function.
    cfg : CfgNode
        Config to pass to the similarity function .

    Returns
    -------
    similarity_fn : object
        The similarity function object.
    """

    if name is None:
        name = cfg.SIMILARITY.NAME

    similarity_fn = SIMILARITY_REGISTRY.get(name)

    return similarity_fn(cfg)
