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


def test_set_criterion_returns_finite_dn_losses():
    criterion = SetCriterion(num_classes=3, matcher=HungarianMatcher())
    logits = torch.randn(2, 12, 3)
    boxes = torch.rand(2, 12, 4)
    dn_logits = torch.randn(2, 8, 3)
    dn_boxes = torch.rand(2, 8, 4)
    targets = [
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
    ]
    dn_meta = {
        "dn_positive_idx": [torch.tensor([0, 1, 4, 5]), torch.tensor([0, 4])],
        "dn_num_group": 2,
        "dn_num_split": [8, 12],
    }
    losses = criterion(
        logits,
        boxes,
        targets,
        dn_outputs={"pred_logits": dn_logits, "pred_boxes": dn_boxes, "aux_outputs": []},
        dn_meta=dn_meta,
    )
    assert all(torch.isfinite(value).all() for value in losses.values())
