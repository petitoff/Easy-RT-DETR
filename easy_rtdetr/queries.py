from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import QueryGroup
from .heads import MLP
from .utils import gather_batch


@dataclass
class QuerySelectionOutput:
    target: torch.Tensor
    reference_points_unact: torch.Tensor
    encoder_boxes: torch.Tensor
    encoder_logits: torch.Tensor
    groups: list[QueryGroup]


class QuerySelection(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_queries: int,
        num_o2o_groups: int,
        o2m_branch: bool,
        num_queries_o2m: int,
        o2m_duplicates: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        self.o2m_duplicates = o2m_duplicates

        total_groups = num_o2o_groups + (1 if o2m_branch else 0)
        self.encoder_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)) for _ in range(total_groups)]
        )
        self.encoder_score_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(total_groups)])
        self.encoder_box_heads = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(total_groups)])
        self.map_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, memory: torch.Tensor, spatial_shapes: torch.Tensor) -> QuerySelectionOutput:
        anchors = self._generate_anchors(spatial_shapes, memory.device, memory.dtype)
        mapped_memory = self.map_memory(memory)
        targets = []
        reference_points_unact = []
        encoder_boxes = []
        encoder_logits = []
        groups: list[QueryGroup] = []

        total_groups = len(self.encoder_proj)
        for group_index in range(total_groups):
            is_primary = group_index == 0
            is_o2m = self.o2m_branch and group_index == total_groups - 1
            query_count = self.num_queries_o2m if is_o2m else self.num_queries
            group_name = "o2m" if is_o2m else f"o2o_{group_index}"
            o2m_duplicates = self.o2m_duplicates if is_o2m else 1
            groups.append(QueryGroup(name=group_name, count=query_count, o2m_duplicates=o2m_duplicates))

            source_memory = memory if is_primary else mapped_memory
            encoded = self.encoder_proj[group_index](source_memory)
            scores = self.encoder_score_heads[group_index](encoded)
            box_unact = self.encoder_box_heads[group_index](encoded) + anchors

            topk = scores.max(dim=-1).values.topk(query_count, dim=1).indices
            targets.append(gather_batch(source_memory, topk))
            reference_points_unact.append(gather_batch(box_unact, topk))
            encoder_boxes.append(gather_batch(box_unact.sigmoid(), topk))
            encoder_logits.append(gather_batch(scores, topk))

        return QuerySelectionOutput(
            target=torch.cat(targets, dim=1),
            reference_points_unact=torch.cat(reference_points_unact, dim=1),
            encoder_boxes=torch.cat(encoder_boxes, dim=1),
            encoder_logits=torch.cat(encoder_logits, dim=1),
            groups=groups,
        )

    def _generate_anchors(self, spatial_shapes: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack((grid_x, grid_y), dim=-1)
            valid_wh = torch.tensor((width, height), device=device, dtype=dtype)
            centers = (grid_xy + 0.5) / valid_wh
            wh = torch.ones_like(centers) * (0.05 * (2.0**level))
            anchors.append(torch.cat((centers, wh), dim=-1).reshape(1, height * width, 4))
        anchors = torch.cat(anchors, dim=1).clamp(min=1e-4, max=1.0 - 1e-4)
        return torch.log(anchors / (1.0 - anchors))
