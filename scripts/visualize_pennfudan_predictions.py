from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.datasets import PennFudanPedDataset, split_indices
from easy_rtdetr.model import RTDETRv3


def cxcywh_to_xyxy(box: torch.Tensor, image_size: int) -> list[float]:
    cx, cy, w, h = box.tolist()
    x0 = (cx - w / 2.0) * image_size
    y0 = (cy - h / 2.0) * image_size
    x1 = (cx + w / 2.0) * image_size
    y1 = (cy + h / 2.0) * image_size
    return [x0, y0, x1, y1]


def build_cpu_model(config_dict: dict | None = None) -> RTDETRv3:
    config = RTDETRv3Config(**config_dict) if config_dict is not None else RTDETRv3Config()
    config.pretrained_backbone = False
    return RTDETRv3(config)


def render_image(
    image_tensor: torch.Tensor,
    target: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    image_size: int,
    score_threshold: float,
    topk: int,
    output_path: Path,
) -> None:
    image = (image_tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)

    for box in target["boxes"]:
        draw.rectangle(cxcywh_to_xyxy(box, image_size), outline=(40, 220, 40), width=3)

    kept = 0
    for score, box in zip(prediction["scores"], prediction["boxes"]):
        if float(score) < score_threshold:
            continue
        draw.rectangle(box.tolist(), outline=(230, 50, 50), width=2)
        draw.text((box[0] + 3, box[1] + 3), f"{float(score):.2f}", fill=(255, 255, 255))
        kept += 1
        if kept >= topk:
            break

    canvas.save(output_path)


def visualize(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = int(checkpoint.get("image_size", args.image_size))
    seed = int(checkpoint.get("seed", args.seed))

    model = build_cpu_model(checkpoint.get("config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.postprocessor.score_threshold = args.score_threshold
    model.postprocessor.nms_threshold = args.nms_threshold
    model.postprocessor.topk = args.topk
    model.eval()

    base_dataset = PennFudanPedDataset(root=args.data_root, image_size=image_size)
    _, eval_indices = split_indices(len(base_dataset), train_fraction=args.train_fraction, seed=seed)
    if args.max_eval_samples is not None:
        eval_indices = eval_indices[: args.max_eval_samples]
    eval_dataset = PennFudanPedDataset(root=args.data_root, image_size=image_size, indices=eval_indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in range(min(args.num_images, len(eval_dataset))):
            image, target = eval_dataset[index]
            prediction = model(image.unsqueeze(0))[0]
            output_path = output_dir / f"pennfudan_pred_{index:02d}.png"
            render_image(
                image_tensor=image,
                target=target,
                prediction=prediction,
                image_size=image_size,
                score_threshold=args.score_threshold,
                topk=args.topk,
                output_path=output_path,
            )
            print(f"saved={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT and predicted boxes for Penn-Fudan checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/pennfudan_cpu_3e.pt")
    parser.add_argument("--data-root", type=str, default="data/PennFudanPed")
    parser.add_argument("--output-dir", type=str, default="artifacts/pennfudan_vis")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-eval-samples", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.18)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
