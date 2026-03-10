import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.queries import QuerySelection


def test_query_selection_shapes():
    selector = QuerySelection(
        hidden_dim=256,
        num_classes=80,
        num_queries=300,
        num_o2o_groups=3,
        o2m_branch=True,
        num_queries_o2m=300,
        o2m_duplicates=4,
        feat_strides=(8, 16, 32),
        learnt_init_query=True,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        anchor_eps=1e-2,
    )
    memory = torch.randn(2, 1344, 256)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]])
    output = selector(memory, spatial_shapes)
    assert output.target.shape == (2, 1200, 256)
    assert output.reference_points_unact.shape == (2, 1200, 4)
    assert output.memory.shape == memory.shape


def test_query_selection_eval_uses_primary_group_only():
    selector = QuerySelection(
        hidden_dim=256,
        num_classes=80,
        num_queries=300,
        num_o2o_groups=3,
        o2m_branch=True,
        num_queries_o2m=300,
        o2m_duplicates=4,
        feat_strides=(8, 16, 32),
        learnt_init_query=True,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        anchor_eps=1e-2,
    )
    selector.eval()
    memory = torch.randn(2, 1344, 256)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]])
    output = selector(memory, spatial_shapes)
    assert output.target.shape == (2, 300, 256)
    assert len(output.groups) == 1
