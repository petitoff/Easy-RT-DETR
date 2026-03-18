from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .attention import MaskedSelfAttention, MultiScaleDeformableAttention
from .heads import MLP
from .utils import get_sine_pos_embed, inverse_sigmoid


@dataclass
class QueryGroup:
    name: str
    count: int
    matching_count: int | None = None
    dn_count: int = 0
    o2m_duplicates: int = 1
    training_only: bool = False
    attn_mask: torch.Tensor | None = field(default=None, repr=False, compare=False)
    dn_meta: dict[str, object] | None = field(default=None, repr=False, compare=False)


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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        attn_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        query = target + query_pos
        target = self.norm1(target + self.dropout(self.self_attn(query, attn_mask)))
        query = target + query_pos
        target = self.norm2(
            target
            + self.dropout(
                self.cross_attn(
                    query=query,
                    reference_points=reference_points,
                    value=memory,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    value_mask=memory_mask,
                )
            )
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
        query_pos_head_inv_sig: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, num_levels, num_points)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.query_pos_head = MLP(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

    def forward(
        self,
        target: torch.Tensor,
        reference_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor | None,
        class_heads: nn.ModuleList,
        box_heads: nn.ModuleList,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = target
        reference_points = reference_points_unact.sigmoid()
        if valid_ratios is None:
            valid_ratios = torch.ones(
                memory.size(0),
                spatial_shapes.size(0),
                2,
                device=memory.device,
                dtype=memory.dtype,
            )
        boxes = []
        logits = []
        for layer_id, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2) * valid_ratios.repeat(1, 1, 2).unsqueeze(1)
            query_pos_source = reference_points_input[..., 0, :]
            if self.query_pos_head_inv_sig:
                query_pos_source = inverse_sigmoid(query_pos_source)
            query_pos = self.query_pos_head(get_sine_pos_embed(query_pos_source, self.hidden_dim // 2))
            output = layer(
                output,
                reference_points_input,
                memory,
                spatial_shapes,
                level_start_index,
                attn_mask,
                memory_mask,
                query_pos,
            )
            norm_output = self.norm(output)
            pred_boxes = (box_heads[layer_id](norm_output) + inverse_sigmoid(reference_points)).sigmoid()
            pred_logits = class_heads[layer_id](norm_output)
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
        block_mask = group.attn_mask
        if block_mask is None:
            block_mask = torch.zeros(group.count, group.count, dtype=torch.bool, device=device)
            should_perturb = index > 0 and not group.training_only
            if should_perturb:
                block_mask = torch.rand(group.count, group.count, device=device) > keep_prob
                block_mask.fill_diagonal_(False)
        else:
            block_mask = block_mask.to(device=device, dtype=torch.bool)
        mask[begin:end, begin:end] = block_mask
        begin = end
    return mask
