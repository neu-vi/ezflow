from ..utils import Registry

ENCODER_REGISTRY = Registry("ENCODER")


def build_encoder(cfg_grp=None, name=None, instantiate=True, **kwargs):

    """
    Build an encoder from a registered encoder name

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Config to pass to the encoder
    name : str
        Name of the registered encoder
    instantiate : bool
        Whether to instantiate the encoder
    kwargs : dict
        Additional keyword arguments to pass to the encoder

    Returns
    -------
    torch.nn.Module
        The encoder object
    """

    if cfg_grp is None:
        assert name is not None, "Must provide name or cfg_grp"
        assert dict(**kwargs) is not None, "Must provide either cfg_grp or kwargs"

    if name is None:
        name = cfg_grp.NAME

    encoder = ENCODER_REGISTRY.get(name)

    if not instantiate:
        return encoder

    if cfg_grp is None:
        return encoder(**kwargs)

    return encoder(cfg_grp, **kwargs)
