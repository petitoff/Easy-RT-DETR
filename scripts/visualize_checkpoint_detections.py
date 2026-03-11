from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.datasets import _letterbox_image
from easy_rtdetr.model import RTDETRv3


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(config_dict: dict | None = None) -> RTDETRv3:
    config = RTDETRv3Config(**config_dict) if config_dict is not None else RTDETRv3Config()
    config.pretrained_backbone = False
    return RTDETRv3(config)


def resolve_inputs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    pattern = "**/*" if recursive else "*"
    images = sorted(candidate for candidate in path.glob(pattern) if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file())
    if not images:
        raise FileNotFoundError(f"No images found under {path}")
    return images


def infer_class_names(raw_names: str | None, num_classes: int) -> list[str]:
    if raw_names:
        names = [name.strip() for name in raw_names.split(",") if name.strip()]
        if len(names) != num_classes:
            raise ValueError(f"Expected {num_classes} class names, got {len(names)}")
        return names
    if num_classes == 1:
        return ["object"]
    return [f"class_{index}" for index in range(num_classes)]


def preprocess_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, float, int, int]:
    resized, resize_info = _letterbox_image(image, image_size, interpolation=InterpolationMode.BILINEAR)
    image_tensor = TF.pil_to_tensor(resized).float() / 255.0
    return image_tensor, resize_info.scale, resize_info.pad_left, resize_info.pad_top


def map_box_to_original(
    box: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
    width: int,
    height: int,
) -> list[float]:
    x0, y0, x1, y1 = box.tolist()
    x0 = (x0 - pad_left) / scale
    y0 = (y0 - pad_top) / scale
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    return [x0, y0, x1, y1]


def draw_predictions(
    image: Image.Image,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    scale: float,
    pad_left: int,
    pad_top: int,
    score_threshold: float,
    topk: int,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    kept = 0
    width, height = image.size
    for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        mapped_box = map_box_to_original(box, scale=scale, pad_left=pad_left, pad_top=pad_top, width=width, height=height)
        if mapped_box[2] <= mapped_box[0] or mapped_box[3] <= mapped_box[1]:
            continue
        label_index = int(label)
        label_name = class_names[label_index] if 0 <= label_index < len(class_names) else f"class_{label_index}"
        draw.rectangle(mapped_box, outline=(230, 50, 50), width=3)
        draw.text((mapped_box[0] + 4, max(0.0, mapped_box[1] - 14)), f"{label_name} {score_value:.2f}", fill=(255, 255, 255))
        kept += 1
        if kept >= topk:
            break
    return canvas


def render_output_path(output_dir: Path, input_root: Path, image_path: Path) -> Path:
    if input_root.is_file():
        return output_dir / f"{image_path.stem}_det{image_path.suffix.lower()}"
    relative_path = image_path.relative_to(input_root)
    return (output_dir / relative_path).with_name(f"{relative_path.stem}_det{relative_path.suffix.lower()}")


def visualize(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = int(checkpoint.get("image_size", args.image_size))

    model = build_model(checkpoint.get("config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.postprocessor.score_threshold = args.score_threshold
    model.postprocessor.nms_threshold = args.nms_threshold
    model.postprocessor.topk = args.topk
    model.eval()

    class_names = infer_class_names(args.class_names, model.config.num_classes)
    input_path = Path(args.input).expanduser().resolve()
    images = resolve_inputs(input_path, recursive=args.recursive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            image_tensor, scale, pad_left, pad_top = preprocess_image(image, image_size=image_size)
            prediction = model(image_tensor.unsqueeze(0))[0]
            rendered = draw_predictions(
                image=image,
                prediction=prediction,
                class_names=class_names,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                score_threshold=args.score_threshold,
                topk=args.topk,
            )
            output_path = render_output_path(output_dir, input_path, image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(output_path)
            print(f"saved={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint inference on images and save visualized detections.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory with images.")
    parser.add_argument("--output-dir", type=str, default="artifacts/inference_vis")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-size", type=int, default=256, help="Used only when checkpoint does not store image_size.")
    parser.add_argument("--score-threshold", type=float, default=0.18)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class names, e.g. 'car' or 'person,bike'.")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
