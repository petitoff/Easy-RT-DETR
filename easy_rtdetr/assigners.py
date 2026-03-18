from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import box_iou


@dataclass(slots=True)
class AssignmentResult:
    labels: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor


def _points_inside_boxes(points: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    x = points[:, 0].unsqueeze(0)
    y = points[:, 1].unsqueeze(0)
    left = x - boxes[:, 0].unsqueeze(1)
    top = y - boxes[:, 1].unsqueeze(1)
    right = boxes[:, 2].unsqueeze(1) - x
    bottom = boxes[:, 3].unsqueeze(1) - y
    deltas = torch.stack((left, top, right, bottom), dim=-1)
    return deltas.min(dim=-1).values > eps


def _empty_assignment(
    batch_size: int,
    num_anchors: int,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AssignmentResult:
    labels = torch.full((batch_size, num_anchors), num_classes, dtype=torch.long, device=device)
    boxes = torch.zeros((batch_size, num_anchors, 4), dtype=dtype, device=device)
    scores = torch.zeros((batch_size, num_anchors, num_classes), dtype=dtype, device=device)
    return AssignmentResult(labels=labels, boxes=boxes, scores=scores)


class ATSSAssigner:
    def __init__(self, topk: int = 9, eps: float = 1e-9) -> None:
        self.topk = topk
        self.eps = eps

    @torch.no_grad()
    def __call__(
        self,
        anchor_boxes: torch.Tensor,
        num_anchors_list: list[int],
        gt_labels: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
        num_classes: int,
        pred_boxes: torch.Tensor | None = None,
    ) -> AssignmentResult:
        batch_size = len(gt_labels)
        num_anchors = anchor_boxes.size(0)
        result = _empty_assignment(batch_size, num_anchors, num_classes, anchor_boxes.device, anchor_boxes.dtype)
        anchor_boxes_float = anchor_boxes.float()
        anchor_centers = (anchor_boxes_float[:, :2] + anchor_boxes_float[:, 2:]) / 2.0
        pred_boxes_float = pred_boxes.float() if pred_boxes is not None else None

        for batch_id, (labels, boxes) in enumerate(zip(gt_labels, gt_boxes)):
            if labels.numel() == 0:
                continue

            boxes_float = boxes.float()
            ious, _ = box_iou(boxes_float, anchor_boxes_float)
            gt_centers = (boxes_float[:, :2] + boxes_float[:, 2:]) / 2.0
            distances = torch.cdist(gt_centers, anchor_centers)
            candidate_mask = torch.zeros_like(ious, dtype=torch.bool)
            candidate_indices: list[torch.Tensor] = []
            start = 0
            for num_level_anchors in num_anchors_list:
                end = start + num_level_anchors
                topk = min(self.topk, num_level_anchors)
                topk_idx = distances[:, start:end].topk(topk, dim=1, largest=False).indices + start
                candidate_indices.append(topk_idx)
                candidate_mask.scatter_(1, topk_idx, True)
                start = end

            flat_candidates = torch.cat(candidate_indices, dim=1)
            candidate_ious = torch.gather(ious, 1, flat_candidates)
            thresholds = candidate_ious.mean(dim=1) + candidate_ious.std(dim=1, unbiased=False)
            positive_mask = candidate_mask & (ious >= thresholds.unsqueeze(1))
            positive_mask &= _points_inside_boxes(anchor_centers, boxes_float)

            if positive_mask.sum() == 0:
                best_anchor = ious.argmax(dim=1, keepdim=True)
                positive_mask.scatter_(1, best_anchor, True)

            positive_ious = ious.masked_fill(~positive_mask, -1.0)
            best_gt_per_anchor = positive_ious.argmax(dim=0)
            positive_anchors = positive_mask.any(dim=0)
            if not positive_anchors.any():
                continue

            pos_idx = positive_anchors.nonzero(as_tuple=False).squeeze(1)
            matched_gt = best_gt_per_anchor[pos_idx]
            result.labels[batch_id, pos_idx] = labels[matched_gt]
            result.boxes[batch_id, pos_idx] = boxes[matched_gt].to(result.boxes.dtype)

            one_hot = F.one_hot(labels[matched_gt], num_classes=num_classes).to(result.scores.dtype)
            if pred_boxes_float is not None:
                pred_ious, _ = box_iou(boxes_float, pred_boxes_float[batch_id])
                matched_scores = pred_ious[matched_gt, pos_idx]
                one_hot = one_hot * matched_scores.unsqueeze(1)
            result.scores[batch_id, pos_idx] = one_hot.to(result.scores.dtype)

        return result


class TaskAlignedAssigner:
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
        num_classes: int,
    ) -> AssignmentResult:
        batch_size, num_anchors, _ = pred_scores.shape
        result = _empty_assignment(batch_size, num_anchors, num_classes, pred_scores.device, pred_scores.dtype)
        pred_scores_float = pred_scores.float()
        pred_boxes_float = pred_boxes.float()
        anchor_points_float = anchor_points.float()

        for batch_id, (labels, boxes) in enumerate(zip(gt_labels, gt_boxes)):
            if labels.numel() == 0:
                continue

            boxes_float = boxes.float()
            ious, _ = box_iou(boxes_float, pred_boxes_float[batch_id])
            cls_scores = pred_scores_float[batch_id][:, labels].transpose(0, 1)
            alignment = cls_scores.pow(self.alpha) * ious.pow(self.beta)
            in_gts = _points_inside_boxes(anchor_points_float, boxes_float)

            topk = min(self.topk, num_anchors)
            topk_idx = (alignment * in_gts.float()).topk(topk, dim=1).indices
            topk_mask = torch.zeros_like(alignment, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)
            positive_mask = topk_mask & in_gts

            if positive_mask.sum() == 0:
                best_anchor = alignment.argmax(dim=1, keepdim=True)
                positive_mask.scatter_(1, best_anchor, True)

            positive_ious = ious.masked_fill(~positive_mask, -1.0)
            best_gt_per_anchor = positive_ious.argmax(dim=0)
            positive_anchors = positive_mask.any(dim=0)
            if not positive_anchors.any():
                continue

            alignment_positive = alignment * positive_mask.float()
            max_metric_per_gt = alignment_positive.max(dim=1).values
            max_iou_per_gt = (ious * positive_mask.float()).max(dim=1).values
            normalized = alignment_positive / (max_metric_per_gt.unsqueeze(1) + self.eps)
            normalized = normalized * max_iou_per_gt.unsqueeze(1)

            pos_idx = positive_anchors.nonzero(as_tuple=False).squeeze(1)
            matched_gt = best_gt_per_anchor[pos_idx]
            result.labels[batch_id, pos_idx] = labels[matched_gt]
            result.boxes[batch_id, pos_idx] = boxes[matched_gt].to(result.boxes.dtype)

            one_hot = F.one_hot(labels[matched_gt], num_classes=num_classes).to(pred_scores.dtype)
            matched_scores = normalized[matched_gt, pos_idx]
            result.scores[batch_id, pos_idx] = (one_hot * matched_scores.unsqueeze(1)).to(result.scores.dtype)

        return result
