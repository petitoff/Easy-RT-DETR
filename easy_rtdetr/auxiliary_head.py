from __future__ import annotations

import torch
from torch import nn


class AuxiliaryDenseHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.box_regressor = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        logits = []
        boxes = []
        locations = []
        for feature in features:
            hidden = self.stem(feature)
            cls = self.classifier(hidden).flatten(2).transpose(1, 2)
            box = self.box_regressor(hidden).sigmoid().flatten(2).transpose(1, 2)
            logits.append(cls)
            boxes.append(box)
            locations.append(self._locations(feature))
        return logits, boxes, locations

    def _locations(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        device = feature.device
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=feature.dtype),
            torch.arange(width, device=device, dtype=feature.dtype),
            indexing="ij",
        )
        return torch.stack(((x + 0.5) / width, (y + 0.5) / height), dim=-1).reshape(-1, 2)
