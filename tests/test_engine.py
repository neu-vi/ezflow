from torch import nn

from ezflow.engine import prune_l1_structured, prune_l1_unstructured


def test_l1_pruning():

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 4),
    )
    _ = prune_l1_structured(model, nn.Linear, 0.5)
    _ = prune_l1_unstructured(model, nn.Linear, 0.5)
