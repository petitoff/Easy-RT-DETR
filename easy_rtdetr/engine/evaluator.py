from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..eval_metrics import compute_detection_map
from ..utils import box_cxcywh_to_xyxy, box_iou


def prediction_metrics(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    match_iou: float,
    duplicate_iou: float,
) -> dict[str, float]:
    metrics = {
        "num_preds": float(pred_boxes.size(0)),
        "num_targets": float(target_boxes.size(0)),
        "best_pred_iou_sum": 0.0,
        "pred_match_count": 0.0,
        "gt_match_count": 0.0,
        "duplicate_pairs": 0.0,
        "total_pairs": 0.0,
    }
    if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
        return metrics
    if pred_boxes.numel() > 0 and target_boxes.numel() > 0:
        pred_to_gt_iou, _ = box_iou(pred_boxes, target_boxes)
        best_pred_iou = pred_to_gt_iou.max(dim=1).values
        best_gt_iou = pred_to_gt_iou.max(dim=0).values
        metrics["best_pred_iou_sum"] = float(best_pred_iou.sum().item())
        metrics["pred_match_count"] = float((best_pred_iou >= match_iou).sum().item())
        metrics["gt_match_count"] = float((best_gt_iou >= match_iou).sum().item())
    if pred_boxes.size(0) >= 2:
        pred_iou, _ = box_iou(pred_boxes, pred_boxes)
        pair_mask = torch.triu(torch.ones_like(pred_iou, dtype=torch.bool), diagonal=1)
        pair_ious = pred_iou[pair_mask]
        metrics["total_pairs"] = float(pair_ious.numel())
        metrics["duplicate_pairs"] = float((pair_ious >= duplicate_iou).sum().item())
    return metrics


@torch.no_grad()
def evaluate_detection_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    evaluation_cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals = {
        "images": 0.0,
        "num_preds": 0.0,
        "num_targets": 0.0,
        "best_pred_iou_sum": 0.0,
        "pred_match_count": 0.0,
        "gt_match_count": 0.0,
        "duplicate_pairs": 0.0,
        "total_pairs": 0.0,
    }
    predictions_for_map: list[dict[str, torch.Tensor]] = []
    targets_for_map: list[dict[str, torch.Tensor]] = []
    match_iou = float(evaluation_cfg.get("match_iou", 0.5))
    duplicate_iou = float(evaluation_cfg.get("duplicate_iou", 0.4))

    for images, targets in data_loader:
        images = images.to(device)
        predictions = model(images)
        for image, target, prediction in zip(images, targets, predictions):
            scale = torch.tensor(
                [image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]],
                dtype=torch.float32,
            )
            target_boxes = box_cxcywh_to_xyxy(target["boxes"]) * scale
            image_metrics = prediction_metrics(
                pred_boxes=prediction["boxes"].cpu(),
                target_boxes=target_boxes.cpu(),
                match_iou=match_iou,
                duplicate_iou=duplicate_iou,
            )
            totals["images"] += 1.0
            for key, value in image_metrics.items():
                totals[key] += value
            predictions_for_map.append(
                {
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu(),
                    "boxes": prediction["boxes"].cpu(),
                }
            )
            targets_for_map.append(
                {
                    "labels": target["labels"].cpu(),
                    "boxes": target_boxes.cpu(),
                }
            )

    image_count = max(totals["images"], 1.0)
    pred_count = max(totals["num_preds"], 1.0)
    target_count = max(totals["num_targets"], 1.0)
    pair_count = max(totals["total_pairs"], 1.0)
    map_result = compute_detection_map(
        predictions=predictions_for_map,
        targets=targets_for_map,
        num_classes=num_classes,
    )
    return {
        "AP50": map_result.ap50,
        "AP75": map_result.ap75,
        "mAP@0.50:0.95": map_result.map,
        "avg_preds": totals["num_preds"] / image_count,
        "avg_targets": totals["num_targets"] / image_count,
        "avg_best_iou": totals["best_pred_iou_sum"] / pred_count,
        f"pred_precision_proxy@{match_iou:.2f}": totals["pred_match_count"] / pred_count,
        f"gt_recall@{match_iou:.2f}": totals["gt_match_count"] / target_count,
        f"duplicate_pair_ratio@{duplicate_iou:.2f}": totals["duplicate_pairs"] / pair_count,
    }
