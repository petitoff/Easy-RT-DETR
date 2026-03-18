from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.datasets import PascalVOCCarDataset, detection_collate_fn, split_indices
from easy_rtdetr.model import RTDETRv3


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_model(args: argparse.Namespace) -> RTDETRv3:
    config = RTDETRv3Config(
        num_classes=1,
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        hidden_dim=args.hidden_dim,
        num_feature_levels=args.num_feature_levels,
        transformer_encoder_layers=args.transformer_encoder_layers,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward=args.dim_feedforward,
        num_o2o_groups=1,
        perturbation_keep_prob=0.9,
        o2m_branch=True,
        num_queries_o2m=args.num_queries_o2m,
        o2m_duplicates=2,
        aux_static_assigner_epoch=4,
        inference_topk=20,
    )
    return RTDETRv3(config)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, PascalVOCCarDataset]:
    root = Path(args.data_root)
    base_dataset = PascalVOCCarDataset(
        root=root,
        image_size=args.image_size,
        split=args.split,
        positive_only=args.positive_only_train,
    )
    train_indices, eval_indices = split_indices(len(base_dataset), train_fraction=args.train_fraction, seed=args.seed)
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_indices = eval_indices[: args.max_eval_samples]

    train_dataset = PascalVOCCarDataset(
        root=root,
        image_size=args.image_size,
        split=args.split,
        indices=train_indices,
        positive_only=args.positive_only_train,
    )
    eval_dataset = PascalVOCCarDataset(
        root=root,
        image_size=args.image_size,
        split=args.split,
        indices=eval_indices,
        positive_only=args.positive_only_train if args.positive_only_eval is None else args.positive_only_eval,
    )
    pin_memory = args.pin_memory and args.device in {"auto", "cuda"}
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_collate_fn,
    )
    return train_loader, eval_dataset


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    train_loader, eval_dataset = make_dataloaders(args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Training on {device.type.upper()} with Pascal VOC car-only detection")
    print(f"data_root={args.data_root}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size}")
    print(
        "model_cfg="
        f"backbone={args.backbone_name} "
        f"hidden_dim={args.hidden_dim} "
        f"num_feature_levels={args.num_feature_levels} "
        f"transformer_encoder_layers={args.transformer_encoder_layers} "
        f"num_decoder_layers={args.num_decoder_layers} "
        f"num_queries={args.num_queries} "
        f"num_queries_o2m={args.num_queries_o2m} "
        f"num_heads={args.num_heads} "
        f"dim_feedforward={args.dim_feedforward}"
    )
    print(
        "runtime_cfg="
        f"num_workers={args.num_workers} "
        f"pin_memory={args.pin_memory} "
        f"amp={use_amp}"
    )
    print(f"train_batches={len(train_loader)} eval_samples={len(eval_dataset)}")

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = [
                {
                    "labels": target["labels"].to(device),
                    "boxes": target["boxes"].to(device),
                }
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                losses = model(images, targets, epoch=epoch + 1)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps += 1
            print(f"epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        print(f"epoch={epoch + 1} avg_loss={running_loss / max(steps, 1):.4f}")

    elapsed = time.time() - started
    print(f"training_time_sec={elapsed:.2f}")

    model.eval()
    image, target = eval_dataset[0]
    with torch.no_grad():
        predictions = model(image.unsqueeze(0).to(device))
    output = predictions[0]
    topk = min(args.preview_topk, output["scores"].numel())
    print("eval_target_boxes=", target["boxes"][:topk].tolist())
    print("pred_scores=", output["scores"][:topk].cpu().tolist())
    print("pred_boxes=", output["boxes"][:topk].cpu().tolist())

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "image_size": args.image_size,
            "seed": args.seed,
            "dataset": "PascalVOC2007Car",
            "split": args.split,
        },
        checkpoint_path,
    )
    print(f"checkpoint_saved={checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETRv3 on Pascal VOC 2007 using only the car class.")
    parser.add_argument("--data-root", type=str, default="data/VOCdevkit/VOC2007")
    parser.add_argument("--split", type=str, default="trainval")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--positive-only-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positive-only-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--preview-topk", type=int, default=5)
    parser.add_argument("--output", type=str, default="artifacts/voc_car_checkpoint.pt")
    parser.add_argument("--backbone-name", type=str, default="resnet34")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-feature-levels", type=int, default=4)
    parser.add_argument("--transformer-encoder-layers", type=int, default=6)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-queries-o2m", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-decoder-points", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=parse_bool, default=True)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
