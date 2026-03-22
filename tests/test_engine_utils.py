import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.engine.evaluator import evaluate_detection_model
from easy_rtdetr.optim import ModelEMA


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def test_model_ema_updates_weights():
    model = _TinyModel()
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    for ema_value, model_value in zip(ema.module.parameters(), model.parameters()):
        assert torch.allclose(ema_value, model_value, atol=1.0)


def test_evaluate_detection_model_returns_map_metrics():
    class DummyDetector(torch.nn.Module):
        def eval(self):
            return self

        def __call__(self, images):
            return [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]]),
                }
                for _ in range(images.size(0))
            ]

    image = torch.zeros(1, 3, 16, 16)
    target = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]])}]
    loader = [(image, target)]
    metrics = evaluate_detection_model(
        DummyDetector(),
        loader,  # type: ignore[arg-type]
        torch.device("cpu"),
        num_classes=1,
        evaluation_cfg={"match_iou": 0.5, "duplicate_iou": 0.4},
    )
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["mAP@0.50:0.95"] == pytest.approx(1.0)
