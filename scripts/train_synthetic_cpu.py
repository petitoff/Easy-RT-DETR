from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from easy_rtdetr.config import RTDETRv3Config
from easy_rtdetr.model import RTDETRv3


class SyntheticDetectionDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        num_classes: int,
        max_objects: int = 3,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.seed = seed
        self.palette = torch.tensor(
            [
                [1.0, 0.2, 0.2],
                [0.2, 1.0, 0.2],
                [0.2, 0.4, 1.0],
                [1.0, 0.8, 0.2],
                [0.8, 0.2, 1.0],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        generator = random.Random(self.seed + index)
        image = torch.rand(3, self.image_size, self.image_size) * 0.05
        num_objects = generator.randint(1, self.max_objects)
        labels = []
        boxes = []

        for _ in range(num_objects):
            label = generator.randrange(self.num_classes)
            width = generator.randint(self.image_size // 8, self.image_size // 3)
            height = generator.randint(self.image_size // 8, self.image_size // 3)
            x0 = generator.randint(0, self.image_size - width - 1)
            y0 = generator.randint(0, self.image_size - height - 1)
            x1 = x0 + width
            y1 = y0 + height

            color = self.palette[label % len(self.palette)].view(3, 1, 1)
            image[:, y0:y1, x0:x1] = color + torch.rand(3, y1 - y0, x1 - x0) * 0.05

            cx = (x0 + x1) / 2.0 / self.image_size
            cy = (y0 + y1) / 2.0 / self.image_size
            bw = (x1 - x0) / self.image_size
            bh = (y1 - y0) / self.image_size
            labels.append(label)
            boxes.append([cx, cy, bw, bh])

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
        return image.clamp(0.0, 1.0), target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    return images, targets


def build_tiny_cpu_model(num_classes: int) -> RTDETRv3:
    config = RTDETRv3Config(
        num_classes=num_classes,
        backbone_name="resnet18",
        pretrained_backbone=False,
        hidden_dim=64,
        num_queries=40,
        num_decoder_layers=2,
        num_heads=4,
        num_decoder_points=4,
        dim_feedforward=128,
        num_o2o_groups=2,
        perturbation_keep_prob=0.9,
        o2m_branch=True,
        num_queries_o2m=40,
        o2m_duplicates=2,
        auxiliary_topk=5,
        auxiliary_hidden_dim=64,
        inference_topk=10,
    )
    return RTDETRv3(config)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")

    train_dataset = SyntheticDetectionDataset(
        num_samples=args.train_samples,
        image_size=args.image_size,
        num_classes=args.num_classes,
        max_objects=args.max_objects,
        seed=args.seed,
    )
    eval_dataset = SyntheticDetectionDataset(
        num_samples=args.eval_samples,
        image_size=args.image_size,
        num_classes=args.num_classes,
        max_objects=args.max_objects,
        seed=args.seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_tiny_cpu_model(args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Training on CPU with synthetic rectangles")
    print(f"train_samples={args.train_samples} eval_samples={args.eval_samples} batch_size={args.batch_size}")
    print(f"image_size={args.image_size} epochs={args.epochs} num_classes={args.num_classes}")

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_steps = 0

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
            optimizer.step()

            running_loss += float(loss.item())
            running_steps += 1
            print(f"epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = running_loss / max(running_steps, 1)
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

    elapsed = time.time() - started
    print(f"training_time_sec={elapsed:.2f}")

    model.eval()
    image, target = eval_dataset[0]
    with torch.no_grad():
        predictions = model(image.unsqueeze(0).to(device))
    output = predictions[0]
    topk = min(args.preview_topk, output["scores"].numel())

    print("eval_target_boxes=", target["boxes"][:topk].tolist())
    print("eval_target_labels=", target["labels"][:topk].tolist())
    print("pred_scores=", output["scores"][:topk].cpu().tolist())
    print("pred_labels=", output["labels"][:topk].cpu().tolist())
    print("pred_boxes=", output["boxes"][:topk].cpu().tolist())

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": args.num_classes,
            "image_size": args.image_size,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    print(f"checkpoint_saved={checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny RT-DETRv3 model on synthetic rectangles using CPU.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-topk", type=int, default=5)
    parser.add_argument("--output", type=str, default="artifacts/synthetic_cpu_checkpoint.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
