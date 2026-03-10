import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from easy_rtdetr.matcher import HungarianMatcher


def test_matcher_handles_empty_targets():
    matcher = HungarianMatcher()
    logits = torch.randn(2, 10, 3)
    boxes = torch.rand(2, 10, 4)
    targets = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
    ]
    indices = matcher(logits, boxes, targets)
    assert len(indices) == 2
    assert indices[0][0].numel() == 0
