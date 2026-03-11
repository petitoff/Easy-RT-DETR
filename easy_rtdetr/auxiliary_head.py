from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .utils import distance_to_bbox


@dataclass(slots=True)
class AuxiliaryHeadOutputs:
    pred_scores: torch.Tensor
    pred_distri: torch.Tensor
    pred_boxes: torch.Tensor
    anchor_boxes: torch.Tensor
    anchor_points: torch.Tensor
    num_anchors_list: list[int]
    stride_tensor: torch.Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ESEAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv = ConvBNAct(channels, channels, kernel_size=1)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feature: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.fc(pooled))
        return self.conv(feature * weight)


class PPYOLOEAuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        fpn_strides: tuple[int, ...],
        reg_max: int = 16,
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fpn_strides = tuple(fpn_strides)
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.reg_channels = reg_max + 1

        self.stem_cls = nn.ModuleList([ESEAttention(in_channels) for _ in self.fpn_strides])
        self.stem_reg = nn.ModuleList([ESEAttention(in_channels) for _ in self.fpn_strides])
        self.pred_cls = nn.ModuleList([nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1) for _ in self.fpn_strides])
        self.pred_reg = nn.ModuleList(
            [nn.Conv2d(in_channels, 4 * self.reg_channels, kernel_size=3, padding=1) for _ in self.fpn_strides]
        )
        self.register_buffer("proj", torch.linspace(0, reg_max, self.reg_channels), persistent=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        prior_prob = 0.01
        bias_cls = float(-torch.log(torch.tensor((1.0 - prior_prob) / prior_prob)))
        for cls_head, reg_head in zip(self.pred_cls, self.pred_reg):
            nn.init.zeros_(cls_head.weight)
            nn.init.constant_(cls_head.bias, bias_cls)
            nn.init.zeros_(reg_head.weight)
            nn.init.ones_(reg_head.bias)

    def _generate_anchors(
        self,
        feats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        anchor_boxes = []
        anchor_points = []
        num_anchors_list: list[int] = []
        stride_tensor = []
        device = feats[0].device
        dtype = feats[0].dtype

        for feat, stride in zip(feats, self.fpn_strides):
            _, _, height, width = feat.shape
            shift_x = (torch.arange(width, device=device, dtype=dtype) + self.grid_cell_offset) * stride
            shift_y = (torch.arange(height, device=device, dtype=dtype) + self.grid_cell_offset) * stride
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            points = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
            half_size = self.grid_cell_scale * stride * 0.5
            boxes = torch.stack(
                (points[:, 0] - half_size, points[:, 1] - half_size, points[:, 0] + half_size, points[:, 1] + half_size),
                dim=-1,
            )
            anchor_boxes.append(boxes)
            anchor_points.append(points)
            num_anchors_list.append(points.size(0))
            stride_tensor.append(torch.full((points.size(0), 1), float(stride), dtype=dtype, device=device))

        return (
            torch.cat(anchor_boxes, dim=0),
            torch.cat(anchor_points, dim=0),
            num_anchors_list,
            torch.cat(stride_tensor, dim=0),
        )

    def decode_boxes(
        self,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        pred_distri: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_anchors, _ = pred_distri.shape
        pred_dist = pred_distri.view(batch_size, num_anchors, 4, self.reg_channels).softmax(dim=-1)
        distances = (pred_dist * self.proj.view(1, 1, 1, -1)).sum(dim=-1) * stride_tensor.unsqueeze(0)
        return distance_to_bbox(anchor_points, distances)

    def forward(self, feats: list[torch.Tensor]) -> AuxiliaryHeadOutputs:
        anchor_boxes, anchor_points, num_anchors_list, stride_tensor = self._generate_anchors(feats)
        cls_score_list = []
        reg_distri_list = []

        for level_id, feat in enumerate(feats):
            pooled = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logits = self.pred_cls[level_id](self.stem_cls[level_id](feat, pooled) + feat)
            reg_distri = self.pred_reg[level_id](self.stem_reg[level_id](feat, pooled))
            cls_score_list.append(cls_logits.sigmoid().flatten(2).transpose(1, 2))
            reg_distri_list.append(reg_distri.flatten(2).transpose(1, 2))

        pred_scores = torch.cat(cls_score_list, dim=1)
        pred_distri = torch.cat(reg_distri_list, dim=1)
        pred_boxes = self.decode_boxes(anchor_points, stride_tensor, pred_distri)
        return AuxiliaryHeadOutputs(
            pred_scores=pred_scores,
            pred_distri=pred_distri,
            pred_boxes=pred_boxes,
            anchor_boxes=anchor_boxes,
            anchor_points=anchor_points,
            num_anchors_list=num_anchors_list,
            stride_tensor=stride_tensor,
        )


AuxiliaryDenseHead = PPYOLOEAuxHead
