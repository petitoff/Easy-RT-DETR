import pytest

torch = pytest.importorskip("torch")

from easy_rtdetr.decoder import QueryGroup, RTDETRDecoder, build_group_attention_mask
from easy_rtdetr.heads import DecoderHeadBundle


def test_decoder_smoke():
    decoder = RTDETRDecoder(
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.0,
        num_levels=3,
        num_points=4,
    )
    heads = DecoderHeadBundle(256, 80, 2)
    target = torch.randn(2, 600, 256)
    reference = torch.rand(2, 600, 4)
    memory = torch.randn(2, 1344, 256)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]])
    level_start_index = torch.tensor([0, 32 * 32, 32 * 32 + 16 * 16])
    valid_ratios = torch.ones(2, 3, 2)
    memory_mask = torch.ones(2, 1344, dtype=torch.bool)
    mask = build_group_attention_mask([QueryGroup("o2o_0", 300), QueryGroup("o2o_1", 300)], 0.9, target.device, True)
    boxes, logits = decoder(
        target,
        reference,
        memory,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        heads.class_heads,
        heads.box_heads,
        mask,
        memory_mask,
    )
    assert boxes.shape == (2, 2, 600, 4)
    assert logits.shape == (2, 2, 600, 80)
