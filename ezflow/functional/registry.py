from ..utils.registry import Registry

FUNCTIONAL_REGISTRY = Registry("FUNCTIONAL")


def get_functional(cfg_grp=None, name=None, **kwargs):
    """
    Retrieve a component from the functional registry

    Parameters
    ----------
    cfg_grp : :class: `CfgNode`
        Configuration for the component
    name : str
        Name of the component
    kwargs : dict
        Additional keyword arguments
    """

    if cfg_grp is None:
        assert name is not None, "Must provide name or cfg_grp"
        assert dict(**kwargs) is not None, "Must provide either cfg_grp or kwargs"

    if name is None:
        name = cfg_grp.NAME

    fn = FUNCTIONAL_REGISTRY.get(name)

    if cfg_grp is None:
        return fn(**kwargs)

    return fn(cfg_grp, **kwargs)
