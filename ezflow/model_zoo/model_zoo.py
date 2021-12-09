"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""


class _ModelZooConfigs:

    MODEL_NAME_TO_CONFIG = {
        "RAFT": "raft.yaml",
        "RAFT_SMALL": "raft_small.yaml",
        "DICL": "dicl.yaml",
        "PWCNet": "pwcnet.yaml",
        "VCN": "vcn.yaml",
        "FlowNetS": "flownet_s.yaml",
        "FlowNetC": "flownet_c.yaml",
    }

    @staticmethod
    def query(model_name):

        if model_name in _ModelZooConfigs.MODEL_NAME_TO_CONFIG:

            cfg_file = _ModelZooConfigs.MODEL_NAME_TO_CONFIG[model_name]
            return cfg_file

        raise ValueError(f"Model name '{model_name}' not found in model zoo")
