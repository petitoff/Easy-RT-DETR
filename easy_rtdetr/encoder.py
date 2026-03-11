from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderOutput:
    features: list[torch.Tensor]
    memory: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor


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


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        use_encoder_idx: tuple[int, ...] = (2,),
        num_encoder_layers: int = 1,
        encoder_num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        pe_temperature: float = 10000.0,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.hidden_dim = hidden_dim
        self.feat_strides = tuple(feat_strides)
        self.use_encoder_idx = tuple(sorted(use_encoder_idx))
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature

        if len(self.in_channels) != len(self.feat_strides):
            raise ValueError("Feature strides must match encoder input features.")
        if any(index < 0 or index >= len(self.in_channels) for index in self.use_encoder_idx):
            raise ValueError("use_encoder_idx contains an invalid feature level.")

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

    def forward(self, features: list[torch.Tensor]) -> EncoderOutput:
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

        flattened = []
        spatial_shapes = []
        level_start_index = [0]
        for feature in pan_features:
            _, _, height, width = feature.shape
            spatial_shapes.append((height, width))
            flattened.append(feature.flatten(2).transpose(1, 2))
            level_start_index.append(level_start_index[-1] + height * width)

        return EncoderOutput(
            features=pan_features,
            memory=torch.cat(flattened, dim=1),
            spatial_shapes=torch.tensor(spatial_shapes, device=pan_features[0].device, dtype=torch.long),
            level_start_index=torch.tensor(level_start_index[:-1], device=pan_features[0].device, dtype=torch.long),
        )
