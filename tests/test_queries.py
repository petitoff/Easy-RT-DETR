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
    memory_mask = torch.ones(2, 1344, dtype=torch.bool)
    output = selector(memory, spatial_shapes, memory_mask=memory_mask)
    assert output.target.shape == (2, 1200, 256)
    assert output.reference_points_unact.shape == (2, 1200, 4)
    assert output.memory.shape == memory.shape
    assert output.memory_mask.shape == (2, 1344)


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


def test_query_selection_anchor_valid_mask_marks_out_of_range_positions():
    selector = QuerySelection(
        hidden_dim=64,
        num_classes=3,
        num_queries=10,
        num_o2o_groups=1,
        o2m_branch=False,
        num_queries_o2m=10,
        o2m_duplicates=1,
        feat_strides=(8,),
        learnt_init_query=True,
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        anchor_eps=0.2,
    )
    anchors, valid_mask = selector._generate_anchors(torch.tensor([[2, 2]]), torch.device("cpu"), torch.float32)
    assert anchors.shape == (1, 4, 4)
    assert valid_mask.shape == (1, 4, 1)
    assert not valid_mask.all()
