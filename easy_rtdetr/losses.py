from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .assigners import ATSSAssigner, TaskAlignedAssigner
from .matcher import HungarianMatcher
from .utils import bbox_to_distance, box_cxcywh_to_xyxy, generalized_box_iou


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
        dn_outputs: dict[str, torch.Tensor] | None = None,
        dn_meta: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        prepared_targets = self._repeat_targets(targets, repeats=o2m_duplicates)
        losses = self._loss_single(pred_logits, pred_boxes, prepared_targets)
        if aux_outputs:
            for layer_id, aux_output in enumerate(aux_outputs):
                aux_loss = self._loss_single(aux_output["pred_logits"], aux_output["pred_boxes"], prepared_targets)
                for key, value in aux_loss.items():
                    losses[f"{key}_aux_{layer_id}"] = value
        if dn_outputs is not None and dn_meta is not None:
            dn_loss = self._loss_denoising(dn_outputs["pred_logits"], dn_outputs["pred_boxes"], targets, dn_meta)
            if "aux_outputs" in dn_outputs:
                for layer_id, aux_output in enumerate(dn_outputs["aux_outputs"]):
                    aux_dn_loss = self._loss_denoising(aux_output["pred_logits"], aux_output["pred_boxes"], targets, dn_meta)
                    for key, value in aux_dn_loss.items():
                        dn_loss[f"{key}_aux_{layer_id}"] = value
            for key, value in dn_loss.items():
                losses[f"{key}_dn"] = value
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

    def _loss_denoising(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        dn_meta: dict[str, object],
    ) -> dict[str, torch.Tensor]:
        num_groups = int(dn_meta["dn_num_group"])
        normalizer = max(float(sum(len(target["labels"]) for target in targets) * num_groups), 1.0)
        target_classes = torch.zeros_like(pred_logits)
        matched_boxes = []
        matched_targets = []

        dn_positive_idx = dn_meta["dn_positive_idx"]
        for batch_id, target in enumerate(targets):
            labels = target["labels"]
            boxes = target["boxes"]
            if labels.numel() == 0:
                continue
            pos_idx = dn_positive_idx[batch_id].to(device=pred_logits.device, dtype=torch.long)
            if pos_idx.numel() == 0:
                continue
            gt_idx = torch.arange(labels.numel(), device=pred_logits.device, dtype=torch.long).repeat(num_groups)
            target_classes[batch_id, pos_idx, labels[gt_idx]] = 1.0
            matched_boxes.append(pred_boxes[batch_id, pos_idx])
            matched_targets.append(boxes[gt_idx])

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
        reg_max: int = 16,
        static_assigner_topk: int = 9,
        task_aligned_topk: int = 13,
        task_aligned_alpha: float = 1.0,
        task_aligned_beta: float = 6.0,
        static_assigner_epoch: int = 30,
        use_varifocal_loss: bool = True,
        loss_weight_class: float = 1.0,
        loss_weight_iou: float = 2.5,
        loss_weight_dfl: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.reg_channels = reg_max + 1
        self.static_assigner_epoch = static_assigner_epoch
        self.use_varifocal_loss = use_varifocal_loss
        self.loss_weight_class = loss_weight_class
        self.loss_weight_iou = loss_weight_iou
        self.loss_weight_dfl = loss_weight_dfl
        self.static_assigner = ATSSAssigner(topk=static_assigner_topk)
        self.task_aligned_assigner = TaskAlignedAssigner(
            topk=task_aligned_topk,
            alpha=task_aligned_alpha,
            beta=task_aligned_beta,
        )

    def forward(
        self,
        aux_outputs,
        targets: list[dict[str, torch.Tensor]],
        image_size: tuple[int, int],
        epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_scores = aux_outputs.pred_scores
        pred_distri = aux_outputs.pred_distri
        pred_boxes = aux_outputs.pred_boxes
        anchor_boxes = aux_outputs.anchor_boxes
        anchor_points = aux_outputs.anchor_points
        stride_tensor = aux_outputs.stride_tensor
        num_anchors_list = aux_outputs.num_anchors_list

        height, width = image_size
        scale = pred_boxes.new_tensor([width, height, width, height])
        gt_labels = [target["labels"].to(device=pred_boxes.device, dtype=torch.long) for target in targets]
        gt_boxes = [box_cxcywh_to_xyxy(target["boxes"].to(pred_boxes.device)) * scale for target in targets]

        if epoch is not None and epoch <= self.static_assigner_epoch:
            assignment = self.static_assigner(
                anchor_boxes=anchor_boxes,
                num_anchors_list=num_anchors_list,
                gt_labels=gt_labels,
                gt_boxes=gt_boxes,
                num_classes=self.num_classes,
                pred_boxes=pred_boxes.detach(),
            )
        else:
            assignment = self.task_aligned_assigner(
                pred_scores=pred_scores.detach(),
                pred_boxes=pred_boxes.detach(),
                anchor_points=anchor_points,
                gt_labels=gt_labels,
                gt_boxes=gt_boxes,
                num_classes=self.num_classes,
            )

        if self.use_varifocal_loss:
            one_hot = F.one_hot(assignment.labels.clamp(max=self.num_classes), self.num_classes + 1)[..., : self.num_classes]
            loss_cls = self._varifocal_loss(pred_scores, assignment.scores, one_hot.to(pred_scores.dtype))
        else:
            target_scores = assignment.scores
            loss_cls = F.binary_cross_entropy(pred_scores, target_scores, reduction="sum")

        assigned_scores_sum = assignment.scores.sum().clamp(min=1.0)
        loss_cls = self.loss_weight_class * (loss_cls / assigned_scores_sum)

        positive_mask = assignment.labels != self.num_classes
        if positive_mask.any():
            bbox_weight = assignment.scores.sum(dim=-1)[positive_mask]
            pred_boxes_pos = pred_boxes[positive_mask]
            assigned_boxes_pos = assignment.boxes[positive_mask]
            loss_iou = 1.0 - torch.diag(generalized_box_iou(pred_boxes_pos, assigned_boxes_pos))
            loss_iou = self.loss_weight_iou * ((loss_iou * bbox_weight).sum() / assigned_scores_sum)

            pred_dist_pos = pred_distri[positive_mask].view(-1, 4, self.reg_channels)
            anchor_points_scaled = anchor_points / stride_tensor
            assigned_boxes_scaled = assignment.boxes / stride_tensor.unsqueeze(0)
            assigned_ltrb = bbox_to_distance(anchor_points_scaled, assigned_boxes_scaled, float(self.reg_max))
            assigned_ltrb_pos = assigned_ltrb[positive_mask]
            loss_dfl = self._distribution_focal_loss(pred_dist_pos, assigned_ltrb_pos)
            loss_dfl = self.loss_weight_dfl * ((loss_dfl * bbox_weight.unsqueeze(-1)).sum() / assigned_scores_sum)
        else:
            zero = pred_boxes.sum() * 0.0
            loss_iou = zero
            loss_dfl = zero

        return {
            "loss_cls": loss_cls,
            "loss_iou": loss_iou,
            "loss_dfl": loss_dfl,
        }

    @staticmethod
    def _varifocal_loss(
        pred_score: torch.Tensor,
        gt_score: torch.Tensor,
        label: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        weight = alpha * pred_score.detach().pow(gamma) * (1.0 - label) + gt_score * label
        return F.binary_cross_entropy(pred_score, gt_score, weight=weight, reduction="sum")

    def _distribution_focal_loss(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_left = target.floor().long()
        target_right = (target_left + 1).clamp(max=self.reg_max)
        weight_left = target_right.to(pred_dist.dtype) - target
        weight_right = 1.0 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_channels),
            target_left.reshape(-1),
            reduction="none",
        ).view_as(target) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_channels),
            target_right.reshape(-1),
            reduction="none",
        ).view_as(target) * weight_right
        return (loss_left + loss_right).mean(dim=-1)
