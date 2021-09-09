import torch

from openoptflow.similarity import LearnableMatchingCost, MutliScalePairwise4DCorr

feature1 = torch.rand(2, 32, 16, 16)
feature1 = torch.rand(2, 32, 16, 16)


def test_LearnableMatchingCost():

    cost_fn = LearnableMatchingCost()
    _ = cost_fn(feature1, feature1)


def test_MultiScalePairwise4DCorr():

    _ = MutliScalePairwise4DCorr(feature1, feature1)
