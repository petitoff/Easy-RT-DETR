import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.auxiliary_head import AuxiliaryDenseHead


def test_auxiliary_head_shapes():
    head = AuxiliaryDenseHead(256, 80, (8, 16, 32))
    features = [torch.randn(2, 256, 32, 32), torch.randn(2, 256, 16, 16), torch.randn(2, 256, 8, 8)]
    outputs = head(features)
    assert outputs.pred_scores.shape == (2, 1024 + 256 + 64, 80)
    assert outputs.pred_distri.shape == (2, 1024 + 256 + 64, 68)
    assert outputs.pred_boxes.shape == (2, 1024 + 256 + 64, 4)
    assert outputs.anchor_points.shape == (1024 + 256 + 64, 2)
