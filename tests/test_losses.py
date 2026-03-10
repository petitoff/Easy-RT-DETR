import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from easy_rtdetr.losses import SetCriterion
from easy_rtdetr.matcher import HungarianMatcher


def test_set_criterion_returns_finite_losses():
    criterion = SetCriterion(num_classes=3, matcher=HungarianMatcher())
    logits = torch.randn(2, 12, 3)
    boxes = torch.rand(2, 12, 4)
    targets = [
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
    ]
    losses = criterion(logits, boxes, targets)
    assert all(torch.isfinite(value).all() for value in losses.values())
