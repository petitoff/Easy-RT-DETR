from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .configuration import ExperimentConfig, load_experiment_config
from .data import build_dataloaders, build_dataset_bundle
from .datasets import _letterbox_image
from .engine import Solver, evaluate_detection_model, load_checkpoint_model
from .engine.solver import resolve_device
from .model import RTDETRv3
from .score_calibration import PrecisionCalibration


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Easy-RT-DETR CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train with a YAML config.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--set", action="append", default=[])

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint with a YAML config.")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--set", action="append", default=[])
    eval_parser.add_argument("--use-raw-model", action="store_true")

    vis_parser = subparsers.add_parser("visualize", help="Visualize detections on images.")
    vis_parser.add_argument("--config", required=True)
    vis_parser.add_argument("--checkpoint", required=True)
    vis_parser.add_argument("--input", required=True)
    vis_parser.add_argument("--output-dir", required=True)
    vis_parser.add_argument("--set", action="append", default=[])
    vis_parser.add_argument("--use-raw-model", action="store_true")
    return parser.parse_args()


def build_model(config: ExperimentConfig) -> RTDETRv3:
    return RTDETRv3(config.build_model_config())


def command_train(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config, overrides=args.set)
    config.data["seed"] = int(config.runtime.get("seed", 0))
    bundle = build_dataset_bundle(config.data)
    train_loader, eval_loader = build_dataloaders(bundle, config.runtime, config.solver)
    solver = Solver(config, build_model(config), train_loader, eval_loader)
    summary = solver.fit()
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_eval(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config, overrides=args.set)
    config.data["seed"] = int(config.runtime.get("seed", 0))
    bundle = build_dataset_bundle(config.data)
    _, eval_loader = build_dataloaders(bundle, config.runtime, config.solver)
    model, checkpoint = load_checkpoint_model(args.checkpoint, use_ema=not args.use_raw_model)
    model.postprocessor.score_threshold = float(config.evaluation.get("score_threshold", 0.18))
    model.postprocessor.nms_threshold = float(config.evaluation.get("nms_threshold", 0.25))
    model.postprocessor.topk = int(config.evaluation.get("topk", 20))
    device = resolve_device(str(config.runtime.get("device", "auto")))
    model.to(device)
    metrics = evaluate_detection_model(
        model,
        eval_loader,
        device,
        num_classes=model.config.num_classes,
        evaluation_cfg=config.evaluation,
    )
    payload = {"checkpoint": str(args.checkpoint), "epoch": checkpoint.get("epoch"), **metrics}
    print(json.dumps(payload, indent=2, sort_keys=True))


def resolve_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    images = sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file())
    if not images:
        raise FileNotFoundError(f"No images found under {path}")
    return images


def preprocess_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, float, int, int]:
    resized, resize_info = _letterbox_image(image, image_size, interpolation=InterpolationMode.BILINEAR)
    image_tensor = TF.pil_to_tensor(resized).float() / 255.0
    return image_tensor, resize_info.scale, resize_info.pad_left, resize_info.pad_top


def map_box_to_original(box: torch.Tensor, scale: float, pad_left: int, pad_top: int, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box.tolist()
    x0 = max(0.0, min(float(width), (x0 - pad_left) / scale))
    y0 = max(0.0, min(float(height), (y0 - pad_top) / scale))
    x1 = max(0.0, min(float(width), (x1 - pad_left) / scale))
    y1 = max(0.0, min(float(height), (y1 - pad_top) / scale))
    return [x0, y0, x1, y1]


def command_visualize(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config, overrides=args.set)
    model, _ = load_checkpoint_model(args.checkpoint, use_ema=not args.use_raw_model)
    model.postprocessor.score_threshold = float(config.evaluation.get("score_threshold", 0.18))
    model.postprocessor.nms_threshold = float(config.evaluation.get("nms_threshold", 0.25))
    model.postprocessor.topk = int(config.evaluation.get("topk", 20))
    device = resolve_device(str(config.runtime.get("device", "auto")))
    model.to(device)
    model.eval()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(config.data.get("classes", [])) or (["object"] if model.config.num_classes == 1 else [f"class_{i}" for i in range(model.config.num_classes)])
    calibration = None
    calibration_path = config.evaluation.get("calibration_path")
    if calibration_path:
        calibration_data = json.loads(Path(str(calibration_path)).read_text(encoding="utf-8"))
        calibration = PrecisionCalibration.from_dict(calibration_data["calibration"])

    for image_path in resolve_inputs(input_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor, scale, pad_left, pad_top = preprocess_image(image, int(config.data["image_size"]))
        with torch.no_grad():
            prediction = model(image_tensor.unsqueeze(0).to(device))[0]
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
            score_value = float(score)
            mapped_box = map_box_to_original(box, scale, pad_left, pad_top, image.width, image.height)
            label_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else f"class_{int(label)}"
            label_text = f"{label_name} score={score_value:.2f}"
            if calibration is not None:
                label_text = f"{label_name} p={calibration.calibrate(score_value):.2f} raw={score_value:.2f}"
            draw.rectangle(mapped_box, outline=(230, 50, 50), width=3)
            draw.text((mapped_box[0] + 4, max(0.0, mapped_box[1] - 14)), label_text, fill=(255, 255, 255))
        if input_path.is_file():
            output_path = output_dir / f"{image_path.stem}_det{image_path.suffix.lower()}"
        else:
            relative_path = image_path.relative_to(input_path)
            output_path = (output_dir / relative_path).with_name(f"{relative_path.stem}_det{relative_path.suffix.lower()}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        print(f"saved={output_path}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        command_train(args)
        return
    if args.command == "eval":
        command_eval(args)
        return
    if args.command == "visualize":
        command_visualize(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
