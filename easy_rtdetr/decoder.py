from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import MaskedSelfAttention, MultiScaleDeformableAttention
from .heads import MLP
from .utils import inverse_sigmoid


@dataclass
class QueryGroup:
    name: str
    count: int
    o2m_duplicates: int = 1


class RTDETRDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.self_attn = MaskedSelfAttention(hidden_dim, num_heads, dropout)
        self.cross_attn = MultiScaleDeformableAttention(hidden_dim, num_heads, num_levels, num_points)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attn_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        query = target + query_pos
        target = self.norm1(target + self.dropout(self.self_attn(query, attn_mask)))
        query = target + query_pos
        target = self.norm2(
            target
            + self.dropout(self.cross_attn(query=query, reference_points=reference_points, value=memory, spatial_shapes=spatial_shapes))
        )
        ffn = self.linear2(self.dropout(torch.relu(self.linear1(target))))
        return self.norm3(target + self.dropout(ffn))


class RTDETRDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, num_levels, num_points)
                for _ in range(num_layers)
            ]
        )
        self.query_pos_head = MLP(4, hidden_dim * 2, hidden_dim, num_layers=2)

    def forward(
        self,
        target: torch.Tensor,
        reference_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        class_heads: nn.ModuleList,
        box_heads: nn.ModuleList,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = target
        reference_points = reference_points_unact.sigmoid()
        boxes = []
        logits = []
        for layer_id, layer in enumerate(self.layers):
            query_pos = self.query_pos_head(inverse_sigmoid(reference_points))
            output = layer(output, reference_points.unsqueeze(2), memory, spatial_shapes, attn_mask, query_pos)
            pred_boxes = (box_heads[layer_id](output) + inverse_sigmoid(reference_points)).sigmoid()
            pred_logits = class_heads[layer_id](output)
            boxes.append(pred_boxes)
            logits.append(pred_logits)
            reference_points = pred_boxes.detach() if self.training else pred_boxes
        return torch.stack(boxes), torch.stack(logits)


def build_group_attention_mask(
    groups: list[QueryGroup],
    keep_prob: float,
    device: torch.device,
    training: bool,
) -> torch.Tensor | None:
    if not training:
        return None

    total_queries = sum(group.count for group in groups)
    mask = torch.zeros(total_queries, total_queries, dtype=torch.bool, device=device)
    begin = 0
    for index, group in enumerate(groups):
        end = begin + group.count
        if index > 0:
            random_mask = torch.rand(group.count, group.count, device=device) > keep_prob
            mask[begin:end, begin:end] = random_mask
        begin = end
    return mask
