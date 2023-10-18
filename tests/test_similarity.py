import torch

from ezflow.similarity import SIMILARITY_REGISTRY

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


def test_MatryoshkaDilatedCostVolume():

    features1 = torch.rand(2, 256, 32, 32)
    features2 = torch.rand(2, 256, 32, 32)
    dilations = [1, 2, 3, 5, 9, 16]

    corr_fn = SIMILARITY_REGISTRY.get("MatryoshkaDilatedCostVolume")(
        max_displacement=4, dilations=dilations
    )

    search_range = corr_fn.get_search_range()
    assert search_range == 9

    offsets = corr_fn.get_relative_offsets()
    assert offsets.shape == (len(dilations), search_range)

    _ = corr_fn(features1, features2)

    corr_fn = SIMILARITY_REGISTRY.get("MatryoshkaDilatedCostVolume")(use_relu=True)
    _ = corr_fn(features1, features2)

    del corr_fn, features1, features2


def test_MatryoshkaDilatedCostVolumeList():

    features1 = [torch.rand(1, 128, 128, 128), torch.rand(1, 256, 32, 32)]
    features2 = [torch.rand(1, 128, 128, 128), torch.rand(1, 256, 32, 32)]

    strides = [2, 8]
    dilations = [[1], [1, 2, 3, 5, 9, 16]]

    corr_fn = SIMILARITY_REGISTRY.get("MatryoshkaDilatedCostVolumeList")(
        max_displacement=4, encoder_output_strides=strides, dilations=dilations
    )

    search_range = corr_fn.get_search_range()
    assert search_range == 9

    offsets = corr_fn.get_global_flow_offsets()
    assert offsets.shape == (
        len(dilations[0]) + len(dilations[1]),
        search_range,
        search_range,
        2,
    )

    _ = corr_fn(features1, features2)

    corr_fn = SIMILARITY_REGISTRY.get("MatryoshkaDilatedCostVolumeList")(
        normalize_feat_l2=True
    )
    _ = corr_fn(features1, features2)

    del corr_fn, features1, features2
