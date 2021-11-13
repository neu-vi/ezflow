import torch

from openoptflow.similarity import (
    Butterfly4D,
    LearnableMatchingCost,
    MutliScalePairwise4DCorr,
    SeparableConv4D,
)

feature1 = torch.rand(2, 32, 16, 16)
feature1 = torch.rand(2, 32, 16, 16)


def test_LearnableMatchingCost():

    similarity_fn = LearnableMatchingCost()
    _ = similarity_fn(feature1, feature1)
    del similarity_fn


def test_MultiScalePairwise4DCorr():

    _ = MutliScalePairwise4DCorr(feature1, feature1)


def test_SeparableConv4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    similarity_fn = SeparableConv4D(2, 4)
    _ = similarity_fn(inp)
    del similarity_fn, inp


def test_Butterfly4D():

    inp = torch.randn(2, 2, 2, 2, 2, 2)
    similarity_fn = Butterfly4D(2, 4)
    _ = similarity_fn(inp)
    del similarity_fn, inp
