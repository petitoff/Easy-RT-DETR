from __future__ import annotations

import copy

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 < len(self.layers):
                x = torch.relu(x)
        return x


class DecoderHeadBundle(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_layers: int) -> None:
        super().__init__()
        cls_head = nn.Linear(hidden_dim, num_classes)
        box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.class_heads = nn.ModuleList([copy.deepcopy(cls_head) for _ in range(num_layers)])
        self.box_heads = nn.ModuleList([copy.deepcopy(box_head) for _ in range(num_layers)])
