import torch
import torch.nn as nn


def replace_bn(module, norm_name):

    if isinstance(module, nn.DataParallel) or isinstance(
        module, nn.parallel.DistributedDataParallel
    ):
        raise Exception("Expected an nn.Module")

    mod = module
    if norm_name == "instance_norm":
        if isinstance(module, nn.BatchNorm2d):
            mod = nn.InstanceNorm2d(
                module.num_features, module.eps, module.momentum, module.affine
            )
        elif isinstance(module, nn.BatchNorm3d):
            mod = nn.InstanceNorm3d(
                module.num_features, module.eps, module.momentum, module.affine
            )
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()
    else:
        raise Exception(f"Not supported norm: {norm_name}")

    for name, child in module.named_children():
        mod.add_module(name, replace_bn(child, norm_name))

    return mod


def replace_relu(module):

    if isinstance(module, nn.DataParallel) or isinstance(
        module, nn.parallel.DistributedDataParallel
    ):
        raise Exception("Expected an nn.Module")

    mod = module
    if isinstance(module, nn.ReLU):
        mod = nn.LeakyReLU(negative_slope=0.1)

    for name, child in module.named_children():
        mod.add_module(name, replace_relu(child))

    return mod
