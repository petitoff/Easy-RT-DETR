from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.datasets import PennFudanPedDataset, detection_collate_fn, split_indices
from easy_rtdetr.model import RTDETRv3


def build_cpu_model(pretrained_backbone: bool = True) -> RTDETRv3:
    config = RTDETRv3Config(
        num_classes=1,
        backbone_name="resnet18",
        pretrained_backbone=pretrained_backbone,
        hidden_dim=96,
        num_queries=60,
        num_decoder_layers=3,
        num_heads=4,
        num_decoder_points=4,
        dim_feedforward=192,
        num_o2o_groups=1,
        perturbation_keep_prob=0.9,
        o2m_branch=True,
        num_queries_o2m=60,
        o2m_duplicates=2,
        auxiliary_topk=5,
        auxiliary_hidden_dim=96,
        inference_topk=10,
    )
    return RTDETRv3(config)


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, PennFudanPedDataset]:
    root = Path(args.data_root)
    base_dataset = PennFudanPedDataset(root=root, image_size=args.image_size)
    train_indices, eval_indices = split_indices(len(base_dataset), train_fraction=args.train_fraction, seed=args.seed)
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_indices = eval_indices[: args.max_eval_samples]

    train_dataset = PennFudanPedDataset(root=root, image_size=args.image_size, indices=train_indices)
    eval_dataset = PennFudanPedDataset(root=root, image_size=args.image_size, indices=eval_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    return train_loader, eval_dataset


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    train_loader, eval_dataset = make_dataloaders(args)
    model = build_cpu_model(pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Training on CPU with Penn-Fudan Pedestrian")
    print(f"data_root={args.data_root}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size}")
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
            losses = model(images, targets)
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            "dataset": "PennFudanPed",
        },
        checkpoint_path,
    )
    print(f"checkpoint_saved={checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny RT-DETRv3 model on Penn-Fudan using CPU.")
    parser.add_argument("--data-root", type=str, default="data/PennFudanPed")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-topk", type=int, default=5)
    parser.add_argument("--output", type=str, default="artifacts/pennfudan_cpu_checkpoint.pt")
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
