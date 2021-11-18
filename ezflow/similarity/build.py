from ..utils import Registry

SIMILARITY_REGISTRY = Registry("SIMILARITY")


def build_similarity(cfg_grp=None, name=None, instantiate=True, **kwargs):

    """
    Build a similarity function from a registered similarity function name.

    Parameters
    ----------
    cfg : CfgNode
        Config to pass to the similarity function.
    name : str
        Name of the registered similarity function.
    instantiate : bool
        Whether to instantiate the similarity function.

    Returns
    -------
    similarity_fn : object
        The similarity function object.
    """

    if cfg_grp is None:
        assert name is not None, "Must provide name or cfg_grp"
        assert dict(**kwargs) is not None, "Must provide either cfg_grp or kwargs"

    if name is None:
        name = cfg_grp.NAME

    similarity_fn = SIMILARITY_REGISTRY.get(name)

    if not instantiate:
        return similarity_fn

    if cfg_grp is None:
        return similarity_fn(**kwargs)

    return similarity_fn(cfg_grp, **kwargs)
