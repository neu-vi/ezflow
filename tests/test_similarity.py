import torch

from openoptflow.similarity import SIMILARITY_REGISTRY

features1 = features2 = torch.rand(2, 32, 16, 16)


def test_CorrelationLayer():

    features1 = torch.rand(2, 8, 32, 32)
    features2 = torch.rand(2, 8, 32, 32)

    corr_fn = SIMILARITY_REGISTRY.get("CorrelationLayer")()
    _ = corr_fn(features1, features2)

    del corr_fn, features1, features2


def test_IterSpatialCorrelationSampler():

    features1 = torch.rand(2, 8, 32, 32)
    features2 = torch.rand(2, 8, 32, 32)

    corr_fn = SIMILARITY_REGISTRY.get("IterSpatialCorrelationSampler")()
    _ = corr_fn(features1, features2)

    del corr_fn, features1, features2


def test_LearnableMatchingCost():

    similarity_fn = SIMILARITY_REGISTRY.get("LearnableMatchingCost")()
    _ = similarity_fn(features1, features2)
    del similarity_fn


def test_MultiScalePairwise4DCorr():

    _ = SIMILARITY_REGISTRY.get("MutliScalePairwise4DCorr")(features1, features2)


def test_SeparableConv4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    similarity_fn = SIMILARITY_REGISTRY.get("SeparableConv4D")(2, 4)
    _ = similarity_fn(inp)

    del similarity_fn, inp


def test_Butterfly4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    similarity_fn = SIMILARITY_REGISTRY.get("Butterfly4D")(2, 4)
    _ = similarity_fn(inp)

    del similarity_fn, inp
