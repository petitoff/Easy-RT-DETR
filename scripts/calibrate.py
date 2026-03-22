from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from easy_rtdetr.configuration import load_experiment_config
from easy_rtdetr.data import build_dataloaders, build_dataset_bundle
from easy_rtdetr.engine import load_checkpoint_model
from easy_rtdetr.engine.solver import resolve_device
from easy_rtdetr.score_calibration import fit_precision_calibration
from easy_rtdetr.utils import box_cxcywh_to_xyxy, box_iou


def collect_detection_records(
    prediction: dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    iou_threshold: float,
) -> list[tuple[float, bool]]:
    scores = prediction["scores"].cpu()
    boxes = prediction["boxes"].cpu()
    if scores.numel() == 0:
        return []
    matched_targets = torch.zeros(target_boxes.size(0), dtype=torch.bool)
    detection_records: list[tuple[float, bool]] = []
    for score, box in zip(scores, boxes):
        is_true_positive = False
        if target_boxes.numel() > 0:
            ious, _ = box_iou(box.unsqueeze(0), target_boxes)
            best_iou, best_target = ious.squeeze(0).max(dim=0)
            if best_iou >= iou_threshold and not matched_targets[best_target]:
                matched_targets[best_target] = True
                is_true_positive = True
        detection_records.append((float(score.item()), is_true_positive))
    return detection_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a precision calibration curve for any Easy-RT-DETR experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--use-raw-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    config.data["seed"] = int(config.runtime.get("seed", 0))
    bundle = build_dataset_bundle(config.data)
    _, eval_loader = build_dataloaders(bundle, config.runtime, config.solver)
    model, _ = load_checkpoint_model(args.checkpoint, use_ema=not args.use_raw_model)
    model.postprocessor.score_threshold = args.score_threshold
    model.postprocessor.nms_threshold = float(config.evaluation.get("nms_threshold", 0.25))
    model.postprocessor.topk = args.topk
    device = resolve_device(str(config.runtime.get("device", "auto")))
    model.to(device).eval()

    detections: list[tuple[float, bool]] = []
    iou_threshold = float(config.evaluation.get("match_iou", 0.5))
    with torch.no_grad():
        for images, targets in eval_loader:
            images = images.to(device)
            predictions = model(images)
            for image, target, prediction in zip(images, targets, predictions):
                scale = torch.tensor([image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]], dtype=torch.float32)
                target_boxes = (box_cxcywh_to_xyxy(target["boxes"]) * scale).cpu()
                detections.extend(collect_detection_records(prediction, target_boxes, iou_threshold))

    calibration = fit_precision_calibration(detections, iou_threshold=iou_threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "calibration": calibration.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
