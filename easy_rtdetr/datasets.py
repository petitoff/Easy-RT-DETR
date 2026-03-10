from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for dataset transforms.") from exc


class PennFudanPedDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        indices: list[int] | None = None,
    ) -> None:
        root = Path(root)
        self.root = root
        self.image_size = image_size
        self.images = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))
        if len(self.images) != len(self.masks):
            raise ValueError("Penn-Fudan image and mask counts do not match.")
        if not self.images:
            raise FileNotFoundError(f"No Penn-Fudan images found under {root}")
        self.indices = indices if indices is not None else list(range(len(self.images)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        index = self.indices[item]
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        image, mask_tensor = self._resize_and_pad(image, mask)
        object_ids = torch.unique(mask_tensor)
        object_ids = object_ids[object_ids != 0]

        boxes = []
        labels = []
        for object_id in object_ids.tolist():
            pos = mask_tensor == object_id
            ys, xs = torch.where(pos)
            if xs.numel() == 0 or ys.numel() == 0:
                continue
            x0 = xs.min().item()
            x1 = xs.max().item() + 1
            y0 = ys.min().item()
            y1 = ys.max().item() + 1
            cx = ((x0 + x1) / 2.0) / self.image_size
            cy = ((y0 + y1) / 2.0) / self.image_size
            bw = (x1 - x0) / self.image_size
            bh = (y1 - y0) / self.image_size
            boxes.append([cx, cy, bw, bh])
            labels.append(0)

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
        return image, target

    def _resize_and_pad(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        orig_width, orig_height = image.size
        scale = min(self.image_size / orig_width, self.image_size / orig_height)
        resized_width = max(1, round(orig_width * scale))
        resized_height = max(1, round(orig_height * scale))

        image = TF.resize(image, [resized_height, resized_width], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [resized_height, resized_width], interpolation=InterpolationMode.NEAREST)

        pad_left = (self.image_size - resized_width) // 2
        pad_top = (self.image_size - resized_height) // 2
        pad_right = self.image_size - resized_width - pad_left
        pad_bottom = self.image_size - resized_height - pad_top
        padding = [pad_left, pad_top, pad_right, pad_bottom]

        image = TF.pad(image, padding, fill=0)
        mask = TF.pad(mask, padding, fill=0)

        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.int64).squeeze(0)
        return image_tensor, mask_tensor


def split_indices(length: int, train_fraction: float = 0.8, seed: int = 0) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    train_length = int(length * train_fraction)
    train_indices = sorted(permutation[:train_length])
    eval_indices = sorted(permutation[train_length:])
    return train_indices, eval_indices


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    return images, targets
