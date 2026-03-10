from __future__ import annotations

import torch
from torch import nn

from .utils import box_cxcywh_to_xyxy, generalized_box_iou

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - runtime fallback only
    linear_sum_assignment = None


class HungarianMatcher(nn.Module):
    def __init__(self, cls_cost: float = 2.0, bbox_cost: float = 5.0, giou_cost: float = 2.0) -> None:
        super().__init__()
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for Hungarian matching.")

        indices = []
        for batch_index, target in enumerate(targets):
            tgt_labels = target["labels"]
            tgt_boxes = target["boxes"]
            if tgt_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            prob = pred_logits[batch_index].sigmoid()
            cost_class = -prob[:, tgt_labels]
            cost_bbox = torch.cdist(pred_boxes[batch_index], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_index]),
                box_cxcywh_to_xyxy(tgt_boxes),
            )
            cost = self.cls_cost * cost_class + self.bbox_cost * cost_bbox + self.giou_cost * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device),
                )
            )
        return indices
