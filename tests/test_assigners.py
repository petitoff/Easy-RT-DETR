import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.assigners import ATSSAssigner, TaskAlignedAssigner


def test_assigners_handle_empty_targets():
    anchor_boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]])
    anchor_points = torch.tensor([[8.0, 8.0], [24.0, 24.0]])
    pred_scores = torch.rand(2, 2, 3)
    pred_boxes = anchor_boxes.unsqueeze(0).repeat(2, 1, 1)
    gt_labels = [torch.empty(0, dtype=torch.long), torch.tensor([1])]
    gt_boxes = [torch.empty(0, 4), torch.tensor([[16.0, 16.0, 32.0, 32.0]])]

    atss = ATSSAssigner(topk=1)
    atss_result = atss(anchor_boxes, [1, 1], gt_labels, gt_boxes, num_classes=3, pred_boxes=pred_boxes)
    assert atss_result.labels.shape == (2, 2)
    assert atss_result.labels[0].eq(3).all()

    task_aligned = TaskAlignedAssigner(topk=1)
    task_result = task_aligned(pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, num_classes=3)
    assert task_result.labels.shape == (2, 2)
    assert task_result.labels[0].eq(3).all()
