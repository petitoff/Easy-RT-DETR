import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.model import RTDETRv3


def test_model_train_and_eval_smoke():
    model = RTDETRv3(RTDETRv3Config(num_classes=3, backbone_name="resnet18", num_queries=50, num_queries_o2m=50))
    images = torch.randn(2, 3, 256, 256)
    targets = [
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)},
    ]
    model.train()
    losses = model(images, targets)
    assert "loss" in losses

    model.eval()
    outputs = model(images)
    assert len(outputs) == 2
    assert {"scores", "labels", "boxes"} <= outputs[0].keys()
