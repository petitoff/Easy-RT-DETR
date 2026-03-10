import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.postprocess import RTDETRPostProcessor


def test_postprocess_scales_boxes():
    processor = RTDETRPostProcessor(topk=5)
    logits = torch.randn(1, 10, 3)
    boxes = torch.rand(1, 10, 4)
    outputs = processor(logits, boxes, image_sizes=torch.tensor([[480, 640]]))
    assert outputs[0]["boxes"].shape == (5, 4)
