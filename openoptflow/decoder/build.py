from ..utils import Registry

DECODER_REGISTRY = Registry("DECODER")


def build_decoder(cfg_grp=None, name=None, instantiate=True, **kwargs):

    """
    Build a decoder from a registered decoder name.

    Parameters
    ----------
    cfg_grp : CfgNode
        Config to pass to the decoder.
    name : str
        Name of the registered decoder.
    instantiate : bool
        Whether to instantiate the decoder.

    Returns
    -------
    decoder : object
        The decoder object.
    """

    if cfg_grp is None:
        assert name is not None, "Must provide name or cfg_grp"
        assert dict(**kwargs) is not None, "Must provide either cfg_grp or kwargs"

    if name is None:
        name = cfg_grp.NAME

    decoder = DECODER_REGISTRY.get(name)

    if not instantiate:
        return decoder

    if cfg_grp is None:
        return decoder(**kwargs)

    return decoder(cfg_grp, **kwargs)
