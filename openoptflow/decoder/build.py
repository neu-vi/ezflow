from ..utils import Registry

DECODER_REGISTRY = Registry("DECODER")


def build_decoder(cfg, name=None):

    """
    Build a decoder from a registered decoder name.

    Parameters
    ----------
    name : str
        Name of the registered decoder.
    cfg : CfgNode
        Config to pass to the decoder.

    Returns
    -------
    decoder : object
        The decoder object.
    """

    if name is None:
        name = cfg.DECODER.NAME

    decoder = DECODER_REGISTRY.get(name)

    return decoder(cfg)