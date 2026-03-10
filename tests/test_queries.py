import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.queries import QuerySelection


def test_query_selection_shapes():
    selector = QuerySelection(
        256,
        num_classes=80,
        num_queries=300,
        num_o2o_groups=3,
        o2m_branch=True,
        num_queries_o2m=300,
        o2m_duplicates=4,
    )
    memory = torch.randn(2, 1344, 256)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]])
    output = selector(memory, spatial_shapes)
    assert output.target.shape == (2, 1200, 256)
    assert output.reference_points_unact.shape == (2, 1200, 4)
