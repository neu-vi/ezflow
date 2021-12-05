import torch
from torch import nn

from ezflow.engine import prune_model


def test_pruning():

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )
    _ = prune_model(model, 0.5)
