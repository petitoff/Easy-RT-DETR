from __future__ import annotations

from typing import Iterable

import torch


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w / 2.0
    half_h = h / 2.0
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    return torch.stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def batch_images(images: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected image tensor shaped [B, C, H, W].")
        return images
    images = list(images)
    if not images:
        raise ValueError("Received an empty image iterable.")
    return torch.stack(images, dim=0)


def gather_batch(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.size(0), device=values.device)[:, None]
    return values[batch, indices]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious, _ = box_iou(boxes[current].unsqueeze(0), boxes[remaining])
        order = remaining[ious[0] <= iou_threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
