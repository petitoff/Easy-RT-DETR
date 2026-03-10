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


class HybridEncoder(nn.Module):
    def __init__(self, in_channels: list[int], hidden_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for channels in in_channels
            ]
        )
        self.lateral_blocks = nn.ModuleList([ConvNormAct(hidden_dim, hidden_dim, 3) for _ in in_channels])
        self.output_blocks = nn.ModuleList([ConvNormAct(hidden_dim, hidden_dim, 3) for _ in in_channels])
        self.downsample_blocks = nn.ModuleList(
            [ConvNormAct(hidden_dim, hidden_dim, 3, stride=2) for _ in range(len(in_channels) - 1)]
        )

    def forward(self, features: list[torch.Tensor]) -> EncoderOutput:
        projected = [proj(feat) for proj, feat in zip(self.input_proj, features)]
        fpn_features = [None] * len(projected)
        fpn_features[-1] = self.lateral_blocks[-1](projected[-1])

        for index in range(len(projected) - 2, -1, -1):
            upsampled = F.interpolate(fpn_features[index + 1], size=projected[index].shape[-2:], mode="nearest")
            fpn_features[index] = self.lateral_blocks[index](projected[index] + upsampled)

        pan_features = [self.output_blocks[0](fpn_features[0])]
        for index in range(1, len(projected)):
            down = self.downsample_blocks[index - 1](pan_features[-1])
            pan_features.append(self.output_blocks[index](fpn_features[index] + down))

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
