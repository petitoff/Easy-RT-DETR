from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import MultiScaleDeformableAttention


@dataclass
class EncoderOutput:
    features: list[torch.Tensor]
    memory: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    valid_ratios: torch.Tensor
    mask_flatten: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RepVggBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(x) + self.conv1(x) + x)


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvNormAct(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[RepVggBlock(hidden_channels) for _ in range(max(1, num_blocks))])
        self.conv3 = (
            ConvNormAct(hidden_channels, out_channels, kernel_size=1)
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.blocks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        q = k = src if pos_embed is None else src + pos_embed
        attn_out, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, pos_embed=pos_embed)
        return src


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, nhead, num_levels, num_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_ffn(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(
        self,
        src: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, query_pos_embed),
            reference_points,
            src,
            spatial_shapes,
            level_start_index=level_start_index,
            value_mask=src_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeformableTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    num_levels=num_levels,
                    num_points=num_points,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.Tensor,
        valid_ratios: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        offset: float = 0.5,
    ) -> torch.Tensor:
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype) + offset,
                torch.arange(width, device=device, dtype=dtype) + offset,
                indexing="ij",
            )
            ref_y = ref_y.reshape(1, -1) / (valid_ratios[:, :, level, 1] * height)
            ref_x = ref_x.reshape(1, -1) / (valid_ratios[:, :, level, 0] * width)
            reference_points.append(torch.stack((ref_x, ref_y), dim=-1))
        reference_points = torch.cat(reference_points, dim=1).unsqueeze(2)
        return reference_points * valid_ratios

    def forward(
        self,
        feat: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        feat_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_ratios is None:
            valid_ratios = torch.ones(
                feat.size(0),
                spatial_shapes.size(0),
                2,
                device=feat.device,
                dtype=feat.dtype,
            )
        reference_points = self.get_reference_points(
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            device=feat.device,
            dtype=feat.dtype,
        )
        for layer in self.layers:
            feat = layer(
                feat,
                reference_points,
                spatial_shapes,
                level_start_index,
                src_mask=feat_mask,
                query_pos_embed=query_pos_embed,
            )
        return feat


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int,
        num_feature_levels: int = 3,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        use_encoder_idx: tuple[int, ...] = (2,),
        num_encoder_layers: int = 1,
        transformer_encoder_layers: int = 1,
        encoder_num_heads: int = 8,
        num_encoder_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        pe_temperature: float = 10000.0,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        use_input_proj: bool = True,
        position_embed_type: str = "sine",
        eval_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.feat_strides = tuple(feat_strides)
        self.use_encoder_idx = tuple(sorted(use_encoder_idx))
        self.num_encoder_layers = num_encoder_layers
        self.transformer_encoder_layers = transformer_encoder_layers
        self.num_encoder_points = num_encoder_points
        self.pe_temperature = pe_temperature
        self.use_input_proj = use_input_proj
        self.position_embed_type = position_embed_type
        self.eval_size = eval_size

        if len(self.in_channels) != len(self.feat_strides):
            raise ValueError("Feature strides must match encoder input features.")
        if any(index < 0 or index >= len(self.in_channels) for index in self.use_encoder_idx):
            raise ValueError("use_encoder_idx contains an invalid feature level.")
        if self.position_embed_type != "sine":
            raise ValueError("Only sine position embedding is currently supported.")

        rep_depth = max(1, round(3 * depth_mult))

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for channels in self.in_channels
            ]
        )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model=hidden_dim,
                    nhead=encoder_num_heads,
                    dim_feedforward=dim_feedforward,
                    num_layers=num_encoder_layers,
                    dropout=dropout,
                )
                for _ in self.use_encoder_idx
            ]
        )

        self.lateral_convs = nn.ModuleList(
            [ConvNormAct(hidden_dim, hidden_dim, kernel_size=1) for _ in range(len(self.in_channels) - 1)]
        )
        self.fpn_blocks = nn.ModuleList(
            [
                CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=rep_depth, expansion=expansion)
                for _ in range(len(self.in_channels) - 1)
            ]
        )
        self.downsample_convs = nn.ModuleList(
            [ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=2) for _ in range(len(self.in_channels) - 1)]
        )
        self.pan_blocks = nn.ModuleList(
            [
                CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=rep_depth, expansion=expansion)
                for _ in range(len(self.in_channels) - 1)
            ]
        )
        self.transformer_input_proj = nn.ModuleList()
        for _ in range(len(self.in_channels)):
            self.transformer_input_proj.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )
        for _ in range(self.num_feature_levels - len(self.in_channels)):
            self.transformer_input_proj.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.deformable_encoder = DeformableTransformerEncoder(
            d_model=hidden_dim,
            nhead=encoder_num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_encoder_layers,
            dropout=dropout,
            num_levels=self.num_feature_levels,
            num_points=num_encoder_points,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.level_embed.weight)
        for module in self.transformer_input_proj:
            nn.init.xavier_uniform_(module[0].weight)

    @staticmethod
    def build_2d_sincos_position_embedding(
        width: int,
        height: int,
        embed_dim: int,
        temperature: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding.")
        grid_w = torch.arange(width, device=device, dtype=dtype)
        grid_h = torch.arange(height, device=device, dtype=dtype)
        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=dtype) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = mesh_w.reshape(-1, 1) * omega.unsqueeze(0)
        out_h = mesh_h.reshape(-1, 1) * omega.unsqueeze(0)
        pos = torch.cat((out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()), dim=1)
        return pos.unsqueeze(0)

    def _apply_encoder(self, projected: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_encoder_layers <= 0 or not self.use_encoder_idx:
            return projected
        projected = [feature for feature in projected]
        for layer_index, feature_index in enumerate(self.use_encoder_idx):
            feature = projected[feature_index]
            batch_size, _, height, width = feature.shape
            src = feature.flatten(2).transpose(1, 2)
            pos_embed = self.build_2d_sincos_position_embedding(
                width=width,
                height=height,
                embed_dim=self.hidden_dim,
                temperature=self.pe_temperature,
                device=feature.device,
                dtype=feature.dtype,
            )
            encoded = self.encoder[layer_index](src, pos_embed=pos_embed)
            projected[feature_index] = encoded.transpose(1, 2).reshape(batch_size, self.hidden_dim, height, width)
        return projected

    @staticmethod
    def _get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        _, height, width = mask.shape
        valid_h = mask[:, :, 0].float().sum(dim=1)
        valid_w = mask[:, 0, :].float().sum(dim=1)
        ratio_h = valid_h / height
        ratio_w = valid_w / width
        return torch.stack((ratio_w, ratio_h), dim=-1)

    def _get_transformer_inputs(
        self,
        features: list[torch.Tensor],
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_input_proj:
            projected = [self.transformer_input_proj[index](feature) for index, feature in enumerate(features)]
            if self.num_feature_levels > len(projected):
                for index in range(len(projected), self.num_feature_levels):
                    source = features[-1] if index == len(features) else projected[-1]
                    projected.append(self.transformer_input_proj[index](source))
        else:
            projected = features

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for index, feature in enumerate(projected):
            batch_size, _, height, width = feature.shape
            spatial_shapes.append((height, width))
            feat_flatten.append(feature.flatten(2).transpose(1, 2))
            if pad_mask is not None:
                level_mask = F.interpolate(
                    pad_mask.unsqueeze(1).float(),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1) > 0.5
            else:
                level_mask = torch.ones(batch_size, height, width, device=feature.device, dtype=torch.bool)
            valid_ratios.append(self._get_valid_ratio(level_mask))
            level_pos_embed = self.build_2d_sincos_position_embedding(
                width=width,
                height=height,
                embed_dim=self.hidden_dim,
                temperature=self.pe_temperature,
                device=feature.device,
                dtype=feature.dtype,
            )
            lvl_pos_embed_flatten.append(level_pos_embed + self.level_embed.weight[index].view(1, 1, -1))
            mask_flatten.append(level_mask.flatten(1))

        spatial_shapes_tensor = torch.tensor(spatial_shapes, device=projected[0].device, dtype=torch.long)
        level_start_index = torch.cat(
            (
                torch.zeros(1, device=projected[0].device, dtype=torch.long),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        return (
            torch.cat(feat_flatten, dim=1),
            spatial_shapes_tensor,
            level_start_index,
            torch.cat(mask_flatten, dim=1),
            torch.cat(lvl_pos_embed_flatten, dim=1),
            torch.stack(valid_ratios, dim=1),
        )

    def forward(self, features: list[torch.Tensor], pad_mask: torch.Tensor | None = None) -> EncoderOutput:
        projected = [proj(feat) for proj, feat in zip(self.input_proj, features)]
        projected = self._apply_encoder(projected)

        inner_outs = [projected[-1]]
        for reverse_index in range(len(self.in_channels) - 1, 0, -1):
            top_index = len(self.in_channels) - 1 - reverse_index
            high_feature = self.lateral_convs[top_index](inner_outs[0])
            low_feature = projected[reverse_index - 1]
            upsampled = F.interpolate(high_feature, size=low_feature.shape[-2:], mode="nearest")
            fused = torch.cat([upsampled, low_feature], dim=1)
            inner_out = self.fpn_blocks[top_index](fused)
            inner_outs[0] = high_feature
            inner_outs.insert(0, inner_out)

        pan_features = [inner_outs[0]]
        for index in range(len(self.in_channels) - 1):
            down = self.downsample_convs[index](pan_features[-1])
            fused = torch.cat([down, inner_outs[index + 1]], dim=1)
            pan_features.append(self.pan_blocks[index](fused))
        (
            feat_flatten,
            spatial_shapes,
            level_start_index,
            mask_flatten,
            lvl_pos_embed_flatten,
            valid_ratios,
        ) = self._get_transformer_inputs(pan_features, pad_mask=pad_mask)
        memory = self.deformable_encoder(
            feat_flatten,
            spatial_shapes,
            level_start_index,
            feat_mask=mask_flatten,
            query_pos_embed=lvl_pos_embed_flatten,
            valid_ratios=valid_ratios,
        )

        return EncoderOutput(
            features=pan_features,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            mask_flatten=mask_flatten,
        )
