import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from easy_rtdetr.backbone import TorchvisionResNetBackbone


def test_backbone_shapes():
    model = TorchvisionResNetBackbone("resnet18", pretrained=False)
    images = torch.randn(2, 3, 256, 256)
    c3, c4, c5 = model(images)
    assert c3.shape[2:] == (32, 32)
    assert c4.shape[2:] == (16, 16)
    assert c5.shape[2:] == (8, 8)
