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


def test_encoder_with_transformer_stage_and_csp_fusion():
    encoder = HybridEncoder(
        [128, 256, 512],
        hidden_dim=128,
        feat_strides=(8, 16, 32),
        use_encoder_idx=(1, 2),
        num_encoder_layers=1,
        encoder_num_heads=4,
        dim_feedforward=256,
    )
    features = [
        torch.randn(1, 128, 32, 32),
        torch.randn(1, 256, 16, 16),
        torch.randn(1, 512, 8, 8),
    ]
    output = encoder(features)
    assert len(output.features) == 3
    assert all(feature.shape[1] == 128 for feature in output.features)
    assert output.level_start_index.tolist() == [0, 32 * 32, 32 * 32 + 16 * 16]
