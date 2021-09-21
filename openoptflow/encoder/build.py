from ..utils import Registry

ENCODER_REGISTRY = Registry("ENCODER")


def build_encoder(cfg, name=None):

    """
    Build an encoder from a registered encoder name.

    Parameters
    ----------
    name : str
        Name of the registered encoder.
    cfg : CfgNode
        Config to pass to the encoder.

    Returns
    -------
    encoder : object
        The encoder object.
    """

    if name is None:
        name = cfg.ENCODER.NAME

    encoder = ENCODER_REGISTRY.get(name)

    return encoder(cfg)
