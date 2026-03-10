import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.auxiliary_head import AuxiliaryDenseHead


def test_auxiliary_head_shapes():
    head = AuxiliaryDenseHead(256, 128, 80)
    features = [torch.randn(2, 256, 32, 32), torch.randn(2, 256, 16, 16), torch.randn(2, 256, 8, 8)]
    logits, boxes, locations = head(features)
    assert len(logits) == 3
    assert logits[0].shape == (2, 1024, 80)
    assert boxes[2].shape == (2, 64, 4)
    assert locations[1].shape == (256, 2)
