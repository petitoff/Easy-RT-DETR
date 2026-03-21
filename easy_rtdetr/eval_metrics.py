from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import box_iou


@dataclass(slots=True)
class DetectionMAPResult:
    map: float
    ap50: float | None
    ap75: float | None
    aps_by_iou: dict[float, float]


def _compute_average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0 or precisions.numel() == 0:
        return 0.0
    mrec = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    changing_points = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    if changing_points.numel() == 0:
        return 0.0
    ap = ((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]).sum()
    return float(ap.item())


def _class_average_precision(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    class_index: int,
    iou_threshold: float,
) -> float | None:
    detections: list[tuple[float, int, torch.Tensor]] = []
    gt_by_image: dict[int, torch.Tensor] = {}
    total_gt = 0

    for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]
        pred_boxes = prediction["boxes"]
        pred_mask = pred_labels == class_index
        for score, box in zip(pred_scores[pred_mask], pred_boxes[pred_mask]):
            detections.append((float(score.item()), image_id, box.detach().cpu()))

        target_labels = target["labels"]
        target_boxes = target["boxes"]
        target_mask = target_labels == class_index
        gt_boxes = target_boxes[target_mask].detach().cpu()
        gt_by_image[image_id] = gt_boxes
        total_gt += int(gt_boxes.size(0))

    if total_gt == 0:
        return None

    detections.sort(key=lambda item: item[0], reverse=True)
    if not detections:
        return 0.0

    matched_gt = {image_id: torch.zeros(len(boxes), dtype=torch.bool) for image_id, boxes in gt_by_image.items()}
    true_positives = torch.zeros(len(detections), dtype=torch.float32)
    false_positives = torch.zeros(len(detections), dtype=torch.float32)

    for detection_index, (_, image_id, pred_box) in enumerate(detections):
        gt_boxes = gt_by_image[image_id]
        if gt_boxes.numel() == 0:
            false_positives[detection_index] = 1.0
            continue

        ious, _ = box_iou(pred_box.unsqueeze(0), gt_boxes)
        best_iou, best_gt_index = ious.squeeze(0).max(dim=0)
        if best_iou >= iou_threshold and not matched_gt[image_id][best_gt_index]:
            true_positives[detection_index] = 1.0
            matched_gt[image_id][best_gt_index] = True
        else:
            false_positives[detection_index] = 1.0

    cum_tp = true_positives.cumsum(dim=0)
    cum_fp = false_positives.cumsum(dim=0)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-8)
    return _compute_average_precision(recalls, precisions)


def compute_detection_map(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: Sequence[float] | None = None,
) -> DetectionMAPResult:
    thresholds = tuple(iou_thresholds) if iou_thresholds is not None else tuple(0.5 + 0.05 * i for i in range(10))
    aps_by_iou: dict[float, float] = {}

    for iou_threshold in thresholds:
        class_aps = []
        for class_index in range(num_classes):
            class_ap = _class_average_precision(predictions, targets, class_index, iou_threshold)
            if class_ap is not None:
                class_aps.append(class_ap)
        aps_by_iou[float(iou_threshold)] = float(sum(class_aps) / len(class_aps)) if class_aps else 0.0

    return DetectionMAPResult(
        map=float(sum(aps_by_iou.values()) / len(aps_by_iou)) if aps_by_iou else 0.0,
        ap50=aps_by_iou.get(0.5),
        ap75=aps_by_iou.get(0.75),
        aps_by_iou=aps_by_iou,
    )
