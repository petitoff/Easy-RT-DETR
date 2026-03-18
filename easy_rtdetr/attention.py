from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.attn(query, query, query, attn_mask=attn_mask, need_weights=False)
        return output


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True).values
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for point in range(self.num_points):
            grid_init[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, len_q, _ = query.shape
        if value_mask is not None:
            value = value.masked_fill(~value_mask.unsqueeze(-1), 0.0)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1).view(
            batch_size, len_q, self.num_heads, self.num_levels, self.num_points
        )

        offset_normalizer = torch.stack((spatial_shapes[:, 1], spatial_shapes[:, 0]), dim=-1).to(query.dtype)
        if reference_points.size(-1) == 2:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.size(-1) == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.size(-1)}")

        value_splits = value.split((spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist(), dim=1)
        sampled_per_level = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            value_level = value_splits[level].permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, height, width)
            sampling_grid = sampling_locations[:, :, :, level].permute(0, 2, 1, 3, 4)
            sampling_grid = sampling_grid.reshape(batch_size * self.num_heads, len_q, self.num_points, 2)
            sampling_grid = sampling_grid * 2.0 - 1.0
            sampled = F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_per_level.append(sampled)

        sampled = torch.stack(sampled_per_level, dim=4)
        sampled = sampled.view(batch_size, self.num_heads, self.head_dim, len_q, self.num_points, self.num_levels)
        sampled = sampled.permute(0, 3, 1, 5, 4, 2)
        weights = attention_weights.unsqueeze(-1)
        output = (sampled * weights).sum(dim=4).sum(dim=3).reshape(batch_size, len_q, self.embed_dim)
        return self.output_proj(output)
