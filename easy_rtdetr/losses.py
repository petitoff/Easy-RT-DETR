from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from .matcher import HungarianMatcher
from .utils import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_factor * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        aux_outputs: list[dict[str, torch.Tensor]] | None = None,
        o2m_duplicates: int = 1,
    ) -> dict[str, torch.Tensor]:
        prepared_targets = self._repeat_targets(targets, repeats=o2m_duplicates)
        losses = self._loss_single(pred_logits, pred_boxes, prepared_targets)
        if aux_outputs:
            for layer_id, aux_output in enumerate(aux_outputs):
                aux_loss = self._loss_single(aux_output["pred_logits"], aux_output["pred_boxes"], prepared_targets)
                for key, value in aux_loss.items():
                    losses[f"{key}_aux_{layer_id}"] = value
        return losses

    def _repeat_targets(self, targets: list[dict[str, torch.Tensor]], repeats: int) -> list[dict[str, torch.Tensor]]:
        if repeats == 1:
            return targets
        expanded = []
        for target in targets:
            expanded.append(
                {
                    "labels": target["labels"].repeat(repeats),
                    "boxes": target["boxes"].repeat(repeats, 1),
                }
            )
        return expanded

    def _loss_single(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(pred_logits, pred_boxes, targets)
        num_boxes = sum(len(target["labels"]) for target in targets)
        normalizer = max(float(num_boxes), 1.0)

        target_classes = torch.zeros_like(pred_logits)
        matched_boxes = []
        matched_targets = []
        for batch_id, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_id, src_idx, targets[batch_id]["labels"][tgt_idx]] = 1.0
            matched_boxes.append(pred_boxes[batch_id, src_idx])
            matched_targets.append(targets[batch_id]["boxes"][tgt_idx])

        loss_cls = sigmoid_focal_loss(
            pred_logits,
            target_classes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / normalizer

        if matched_boxes:
            src_boxes = torch.cat(matched_boxes, dim=0)
            tgt_boxes = torch.cat(matched_targets, dim=0)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / normalizer
            loss_giou = (1.0 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))).sum()
            loss_giou = loss_giou / normalizer
        else:
            zero = pred_boxes.sum() * 0.0
            loss_bbox = zero
            loss_giou = zero

        return {
            "loss_cls": loss_cls * self.cls_weight,
            "loss_bbox": loss_bbox * self.bbox_weight,
            "loss_giou": loss_giou * self.giou_weight,
        }


class AuxiliaryDenseCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        topk: int = 9,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_logits: list[torch.Tensor],
        pred_boxes: list[torch.Tensor],
        locations: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        all_logits = torch.cat(pred_logits, dim=1)
        all_boxes = torch.cat(pred_boxes, dim=1)
        all_locations = torch.cat(locations, dim=0)
        target_classes = torch.zeros_like(all_logits)
        positives: dict[int, list[int]] = defaultdict(list)
        target_boxes: dict[int, list[torch.Tensor]] = defaultdict(list)

        for batch_index, target in enumerate(targets):
            if target["labels"].numel() == 0:
                continue
            gt_centers = target["boxes"][:, :2]
            distances = torch.cdist(all_locations, gt_centers)
            k = min(self.topk, all_locations.size(0))
            indices = distances.topk(k, largest=False, dim=0).indices
            for gt_index in range(indices.size(1)):
                pos_idx = indices[:, gt_index]
                positives[batch_index].extend(pos_idx.tolist())
                target_classes[batch_index, pos_idx, target["labels"][gt_index]] = 1.0
                target_boxes[batch_index].append(target["boxes"][gt_index].repeat(pos_idx.numel(), 1))

        loss_cls = sigmoid_focal_loss(
            all_logits,
            target_classes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        box_losses = []
        giou_losses = []
        for batch_index, positive_indices in positives.items():
            if not positive_indices:
                continue
            pos_idx = torch.as_tensor(positive_indices, device=all_boxes.device, dtype=torch.long)
            pred = all_boxes[batch_index, pos_idx]
            tgt = torch.cat(target_boxes[batch_index], dim=0)
            box_losses.append(F.l1_loss(pred, tgt, reduction="mean"))
            giou_losses.append(
                (1.0 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred), box_cxcywh_to_xyxy(tgt)))).mean()
            )

        zero = all_boxes.sum() * 0.0
        loss_bbox = torch.stack(box_losses).mean() if box_losses else zero
        loss_giou = torch.stack(giou_losses).mean() if giou_losses else zero
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
