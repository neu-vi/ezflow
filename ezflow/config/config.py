"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

import functools
import inspect

from fvcore.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    Container class for configs
    """

    def to_dict(self, cfg_node=None, key_list=[]):

        """Convert a config node to dictionary"""

        VALID_TYPES = {tuple, list, str, int, float, bool}

        if cfg_node is None:
            cfg_node = self

        if not isinstance(cfg_node, CfgNode):

            if type(cfg_node) not in VALID_TYPES:
                print(
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list), type(cfg_node), VALID_TYPES
                    ),
                )

            return cfg_node

        else:

            cfg_dict = dict(cfg_node)

            for k, v in cfg_dict.items():
                cfg_dict[k] = self.to_dict(v, key_list + [k])

            return cfg_dict

    def dump(self, *args, **kwargs):
        return super().dump(*args, **kwargs)


def configurable(init_func=None, *, from_config=None):

    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.

    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):

            try:
                from_config_func = type(self).from_config

            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e

            if not inspect.ismethod(from_config_func):
                raise TypeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                )

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:

        if from_config is None:
            return configurable

        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):

                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)

                    return orig_func(**explicit_args)

                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config

            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):

    signature = inspect.signature(from_config_func)

    if list(signature.parameters.keys())[0] != "cfg":

        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__

        else:
            name = f"{from_config_func.__self__}.from_config"

        raise TypeError(f"{name} must take 'cfg' as the first argument!")

    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )

    if support_var_arg:
        ret = from_config_func(*args, **kwargs)

    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}

        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)

        ret = from_config_func(*args, **kwargs)

        ret.update(extra_kwargs)

    return ret


def _called_with_cfg(*args, **kwargs):

    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (_CfgNode, DictConfig)):
        return True

    return False
