import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.eval_metrics import compute_detection_map


def test_compute_detection_map_returns_perfect_score_for_perfect_detection():
    predictions = [
        {
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        }
    ]
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        }
    ]

    result = compute_detection_map(predictions, targets, num_classes=1, iou_thresholds=[0.5, 0.75])
    assert result.ap50 == pytest.approx(1.0)
    assert result.ap75 == pytest.approx(1.0)
    assert result.map == pytest.approx(1.0)


def test_compute_detection_map_penalizes_false_positive_before_true_positive():
    predictions = [
        {
            "scores": torch.tensor([0.99, 0.80]),
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor(
                [
                    [20.0, 20.0, 30.0, 30.0],
                    [0.0, 0.0, 10.0, 10.0],
                ]
            ),
        }
    ]
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        }
    ]

    result = compute_detection_map(predictions, targets, num_classes=1, iou_thresholds=[0.5])
    assert result.ap50 == pytest.approx(0.5)
    assert result.map == pytest.approx(0.5)


def test_compute_detection_map_averages_iou_thresholds():
    predictions = [
        {
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        }
    ]
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[1.0, 1.0, 11.0, 11.0]]),
        }
    ]

    result = compute_detection_map(predictions, targets, num_classes=1, iou_thresholds=[0.5, 0.75])
    assert result.ap50 == pytest.approx(1.0)
    assert result.ap75 == pytest.approx(0.0)
    assert result.map == pytest.approx(0.5)
