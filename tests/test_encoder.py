import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.encoder import HybridEncoder


def test_encoder_flattened_memory():
    encoder = HybridEncoder([128, 256, 512], hidden_dim=256)
    features = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    output = encoder(features)
    assert output.memory.shape == (2, 32 * 32 + 16 * 16 + 8 * 8, 256)
    assert output.spatial_shapes.shape == (3, 2)
