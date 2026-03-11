from __future__ import annotations

import argparse

import torch

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.datasets import PascalVOCCarDataset, split_indices
from easy_rtdetr.model import RTDETRv3
from easy_rtdetr.utils import box_cxcywh_to_xyxy, box_iou


def build_model(config_dict: dict | None = None) -> RTDETRv3:
    config = RTDETRv3Config(**config_dict) if config_dict is not None else RTDETRv3Config()
    config.pretrained_backbone = False
    return RTDETRv3(config)


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


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", args.image_size))
    seed = int(checkpoint.get("seed", args.seed))

    model = build_model(checkpoint.get("config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.postprocessor.score_threshold = args.score_threshold
    model.postprocessor.nms_threshold = args.nms_threshold
    model.postprocessor.topk = args.topk
    model.eval()

    base_dataset = PascalVOCCarDataset(
        root=args.data_root,
        image_size=image_size,
        split=args.split,
        positive_only=args.positive_only_eval,
    )
    _, eval_indices = split_indices(len(base_dataset), train_fraction=args.train_fraction, seed=seed)
    if args.max_eval_samples is not None:
        eval_indices = eval_indices[: args.max_eval_samples]
    eval_dataset = PascalVOCCarDataset(
        root=args.data_root,
        image_size=image_size,
        split=args.split,
        indices=eval_indices,
        positive_only=args.positive_only_eval,
    )

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

    with torch.no_grad():
        for image, target in eval_dataset:
            prediction = model(image.unsqueeze(0))[0]
            target_boxes = box_cxcywh_to_xyxy(target["boxes"]) * image_size
            image_metrics = prediction_metrics(
                pred_boxes=prediction["boxes"],
                target_boxes=target_boxes,
                match_iou=args.match_iou,
                duplicate_iou=args.duplicate_iou,
            )
            totals["images"] += 1.0
            for key, value in image_metrics.items():
                totals[key] += value

    image_count = max(totals["images"], 1.0)
    pred_count = max(totals["num_preds"], 1.0)
    target_count = max(totals["num_targets"], 1.0)
    pair_count = max(totals["total_pairs"], 1.0)

    print(f"checkpoint={checkpoint_path}")
    print(f"image_size={image_size} eval_images={int(totals['images'])}")
    print(f"score_threshold={args.score_threshold:.2f} nms_threshold={args.nms_threshold:.2f} topk={args.topk}")
    print(f"avg_preds={totals['num_preds'] / image_count:.3f}")
    print(f"avg_targets={totals['num_targets'] / image_count:.3f}")
    print(f"avg_best_iou={totals['best_pred_iou_sum'] / pred_count:.3f}")
    print(f"pred_precision_proxy@{args.match_iou:.2f}={totals['pred_match_count'] / pred_count:.3f}")
    print(f"gt_recall@{args.match_iou:.2f}={totals['gt_match_count'] / target_count:.3f}")
    print(f"duplicate_pair_ratio@{args.duplicate_iou:.2f}={totals['duplicate_pairs'] / pair_count:.3f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pascal VOC car-only checkpoints with simple detection metrics.")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data-root", type=str, default="data/VOCdevkit/VOC2007")
    parser.add_argument("--split", type=str, default="trainval")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--positive-only-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.18)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--duplicate-iou", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for checkpoint_path in arguments.checkpoint:
        evaluate_checkpoint(arguments, checkpoint_path)
