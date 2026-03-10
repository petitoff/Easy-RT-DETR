from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import QueryGroup
from .denoising import get_contrastive_denoising_training_group
from .heads import MLP
from .utils import gather_batch


@dataclass
class QuerySelectionOutput:
    target: torch.Tensor
    reference_points_unact: torch.Tensor
    encoder_boxes: torch.Tensor
    encoder_logits: torch.Tensor
    groups: list[QueryGroup]
    memory: torch.Tensor


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
        feat_strides: tuple[int, ...],
        learnt_init_query: bool,
        num_denoising: int,
        label_noise_ratio: float,
        box_noise_scale: float,
        anchor_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_o2o_groups = num_o2o_groups
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        self.o2m_duplicates = o2m_duplicates
        self.feat_strides = feat_strides
        self.learnt_init_query = learnt_init_query
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.anchor_eps = anchor_eps

        total_groups = num_o2o_groups + (1 if o2m_branch else 0)
        self.encoder_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)) for _ in range(total_groups)]
        )
        self.encoder_score_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(total_groups)])
        self.encoder_box_heads = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(total_groups)])
        self.map_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.max_query_count = max(num_queries, num_queries_o2m)
        self.tgt_embed = nn.Embedding(self.max_query_count, hidden_dim)
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias_cls = -torch.log(torch.tensor((1.0 - 0.01) / 0.01)).item()
        for encoder_proj, score_head, box_head in zip(self.encoder_proj, self.encoder_score_heads, self.encoder_box_heads):
            nn.init.xavier_uniform_(encoder_proj[0].weight)
            nn.init.constant_(encoder_proj[0].bias, 0.0)
            nn.init.xavier_uniform_(score_head.weight)
            nn.init.constant_(score_head.bias, bias_cls)
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)
        nn.init.xavier_uniform_(self.map_memory[0].weight)
        nn.init.constant_(self.map_memory[0].bias, 0.0)
        nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.normal_(self.denoising_class_embed.weight, std=1.0)

    def forward(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> QuerySelectionOutput:
        anchors, valid_mask = self._generate_anchors(spatial_shapes, memory.device, memory.dtype)
        masked_memory = torch.where(valid_mask, memory, torch.zeros_like(memory))
        mapped_memory = self.map_memory(masked_memory.detach())
        targets_out = []
        reference_points_unact = []
        encoder_boxes = []
        encoder_logits = []
        groups: list[QueryGroup] = []

        total_groups = len(self.encoder_proj)
        active_groups = total_groups if self.training else 1
        for group_index in range(active_groups):
            is_primary = group_index == 0
            is_o2m = self.o2m_branch and group_index == total_groups - 1 and self.training
            query_count = self.num_queries_o2m if is_o2m else self.num_queries
            group_name = "o2m" if is_o2m else f"o2o_{group_index}"
            encoded = self.encoder_proj[group_index](masked_memory)
            scores = self.encoder_score_heads[group_index](encoded)
            box_unact = self.encoder_box_heads[group_index](encoded) + anchors

            topk = scores.max(dim=-1).values.topk(query_count, dim=1).indices
            encoder_boxes.append(gather_batch(box_unact.sigmoid(), topk))
            encoder_logits.append(gather_batch(scores, topk))
            if self.learnt_init_query:
                group_target = self.tgt_embed.weight[:query_count].unsqueeze(0).expand(memory.size(0), -1, -1)
            else:
                source_memory = encoded if is_primary else mapped_memory
                group_target = gather_batch(source_memory, topk)
            group_reference_points = gather_batch(box_unact, topk)

            dn_count = 0
            dn_meta = None
            local_attn_mask = None
            if self.training and targets is not None and not is_o2m:
                denoising = get_contrastive_denoising_training_group(
                    targets=targets,
                    num_classes=self.num_classes,
                    num_queries=query_count,
                    class_embed=self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
                if denoising is not None:
                    group_target = torch.cat((denoising.query_class, group_target), dim=1)
                    group_reference_points = torch.cat((denoising.query_bbox_unact, group_reference_points), dim=1)
                    dn_count = denoising.query_bbox_unact.size(1)
                    dn_meta = denoising.dn_meta
                    local_attn_mask = denoising.attn_mask

            if self.training:
                group_reference_points = group_reference_points.detach()

            targets_out.append(group_target)
            reference_points_unact.append(group_reference_points)
            groups.append(
                QueryGroup(
                    name=group_name,
                    count=group_target.size(1),
                    matching_count=query_count,
                    dn_count=dn_count,
                    o2m_duplicates=self.o2m_duplicates if is_o2m else 1,
                    training_only=is_o2m,
                    attn_mask=local_attn_mask,
                    dn_meta=dn_meta,
                )
            )

        return QuerySelectionOutput(
            target=torch.cat(targets_out, dim=1),
            reference_points_unact=torch.cat(reference_points_unact, dim=1),
            encoder_boxes=torch.cat(encoder_boxes, dim=1),
            encoder_logits=torch.cat(encoder_logits, dim=1),
            groups=groups,
            memory=masked_memory,
        )

    def _generate_anchors(
        self,
        spatial_shapes: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = []
        valid_masks = []
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
            anchor = torch.cat((centers, wh), dim=-1).reshape(1, height * width, 4)
            valid_mask = ((anchor > self.anchor_eps) & (anchor < 1.0 - self.anchor_eps)).all(dim=-1, keepdim=True)
            anchors.append(torch.log(anchor.clamp(min=1e-4, max=1.0 - 1e-4) / (1.0 - anchor.clamp(min=1e-4, max=1.0 - 1e-4))))
            valid_masks.append(valid_mask)
        return torch.cat(anchors, dim=1), torch.cat(valid_masks, dim=1)
