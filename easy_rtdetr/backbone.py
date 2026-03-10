from __future__ import annotations

import torch
from torch import nn


_CHANNELS = {
    "resnet18": (128, 256, 512),
    "resnet34": (128, 256, 512),
    "resnet50": (512, 1024, 2048),
    "resnet101": (512, 1024, 2048),
}


class TorchvisionResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        if name not in _CHANNELS:
            raise ValueError(f"Unsupported backbone: {name}")

        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("torchvision is required to build the backbone.") from exc

        weights = models.get_model_weights(name).DEFAULT if pretrained else None
        backbone = getattr(models, name)(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = list(_CHANNELS[name])
        self.out_strides = [8, 16, 32]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]
