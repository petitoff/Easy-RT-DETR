from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for dataset transforms.") from exc


@dataclass(slots=True)
class ResizeInfo:
    scale: float
    pad_left: int
    pad_top: int


def _letterbox_image(
    image: Image.Image,
    image_size: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> tuple[Image.Image, ResizeInfo]:
    orig_width, orig_height = image.size
    scale = min(image_size / orig_width, image_size / orig_height)
    resized_width = max(1, round(orig_width * scale))
    resized_height = max(1, round(orig_height * scale))

    resized = TF.resize(image, [resized_height, resized_width], interpolation=interpolation)
    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    pad_right = image_size - resized_width - pad_left
    pad_bottom = image_size - resized_height - pad_top
    padded = TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    return padded, ResizeInfo(scale=scale, pad_left=pad_left, pad_top=pad_top)


def _resize_xyxy_boxes(boxes: torch.Tensor, resize_info: ResizeInfo, image_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_info.scale + resize_info.pad_left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_info.scale + resize_info.pad_top
    boxes = boxes.clamp(min=0.0, max=float(image_size))
    return boxes


def _xyxy_to_normalized_cxcywh(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    cx = ((x0 + x1) / 2.0) / image_size
    cy = ((y0 + y1) / 2.0) / image_size
    w = (x1 - x0) / image_size
    h = (y1 - y0) / image_size
    return torch.stack((cx, cy, w, h), dim=-1)


BDD100K_VEHICLE_CLASSES = ("car", "truck", "bus")


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
            boxes.append([x0, y0, x1, y1])
            labels.append(0)

        boxes_tensor = _xyxy_to_normalized_cxcywh(torch.tensor(boxes, dtype=torch.float32), self.image_size)
        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": boxes_tensor,
        }
        return image, target

    def _resize_and_pad(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = _letterbox_image(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask, _ = _letterbox_image(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.int64).squeeze(0)
        return image_tensor, mask_tensor


class PascalVOCCarDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        split: str = "trainval",
        indices: list[int] | None = None,
        keep_difficult: bool = False,
        positive_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.keep_difficult = keep_difficult
        split_path = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"VOC split file not found: {split_path}")
        identifiers = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not identifiers:
            raise FileNotFoundError(f"No image ids found in split: {split_path}")
        if positive_only:
            identifiers = [
                image_id
                for image_id in identifiers
                if self._annotation_has_car(self.root / "Annotations" / f"{image_id}.xml")
            ]
            if not identifiers:
                raise FileNotFoundError(f"No positive car samples found in split: {split_path}")
        self.identifiers = identifiers if indices is None else [identifiers[index] for index in indices]

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.identifiers[item]
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = self.root / "Annotations" / f"{image_id}.xml"
        image = Image.open(image_path).convert("RGB")
        boxes_xyxy = self._load_car_boxes(annotation_path)
        image, resize_info = _letterbox_image(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        boxes_xyxy = _resize_xyxy_boxes(boxes_xyxy, resize_info, self.image_size)
        boxes = _xyxy_to_normalized_cxcywh(boxes_xyxy, self.image_size)
        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        target = {
            "labels": torch.zeros(boxes.size(0), dtype=torch.long),
            "boxes": boxes.to(dtype=torch.float32),
        }
        return image_tensor, target

    def _load_car_boxes(self, annotation_path: Path) -> torch.Tensor:
        root = ET.parse(annotation_path).getroot()
        boxes: list[list[float]] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if name != "car":
                continue
            difficult = int(obj.findtext("difficult", default="0"))
            if difficult and not self.keep_difficult:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            x0 = float(bbox.findtext("xmin", default="0")) - 1.0
            y0 = float(bbox.findtext("ymin", default="0")) - 1.0
            x1 = float(bbox.findtext("xmax", default="0"))
            y1 = float(bbox.findtext("ymax", default="0"))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def _annotation_has_car(self, annotation_path: Path) -> bool:
        return bool(self._load_car_boxes(annotation_path).numel())


class BDD100KVehicleDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        split: str = "train",
        indices: list[int] | None = None,
        classes: tuple[str, ...] = BDD100K_VEHICLE_CLASSES,
        positive_only: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.split = split
        self.class_names = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.positive_only = positive_only

        label_file_candidates = [
            self.root / "labels" / f"det_{split}_vehicle3.json",
            self.root / "labels" / f"det_{split}.json",
        ]
        label_dir_candidates = [
            self.root / "labels" / "100k" / split,
            self.root / "100k" / split,
        ]
        label_path = next((candidate for candidate in label_file_candidates if candidate.exists()), None)
        label_dir = next((candidate for candidate in label_dir_candidates if candidate.exists()), None)
        if label_path is None and label_dir is None:
            raise FileNotFoundError(f"BDD100K labels not found for split={split}")

        image_dir_candidates = [
            self.root / "images" / split,
            self.root / "images" / "100k" / split,
            self.root / "10k" / split,
        ]
        image_dir = next((candidate for candidate in image_dir_candidates if candidate.exists()), None)
        if image_dir is None:
            raise FileNotFoundError(f"BDD100K image directory not found for split={split}: {image_dir_candidates[0]}")

        samples: list[dict[str, object]] = []
        if label_path is not None:
            entries = json.loads(label_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                raise ValueError(f"Unexpected BDD100K label format in {label_path}")
            for entry in entries:
                sample = self._sample_from_aggregate_entry(entry, image_dir)
                if sample is None:
                    continue
                if positive_only and not sample["boxes"]:
                    continue
                samples.append(sample)
        else:
            assert label_dir is not None
            for annotation_path in sorted(label_dir.glob("*.json")):
                sample = self._sample_from_frame_file(annotation_path, image_dir)
                if sample is None:
                    continue
                if positive_only and not sample["boxes"]:
                    continue
                samples.append(sample)

        if not samples:
            raise FileNotFoundError(f"No BDD100K samples found under {self.root} for split={split}")
        self.samples = samples if indices is None else [samples[index] for index in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[item]
        image = Image.open(sample["image_path"]).convert("RGB")
        boxes_xyxy = torch.tensor(sample["boxes"], dtype=torch.float32) if sample["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.long) if sample["labels"] else torch.zeros((0,), dtype=torch.long)
        image, resize_info = _letterbox_image(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        boxes_xyxy = _resize_xyxy_boxes(boxes_xyxy, resize_info, self.image_size)
        boxes = _xyxy_to_normalized_cxcywh(boxes_xyxy, self.image_size)
        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        target = {
            "labels": labels,
            "boxes": boxes.to(dtype=torch.float32),
        }
        return image_tensor, target

    def _parse_vehicle_annotations(self, raw_labels: object) -> tuple[list[list[float]], list[int]]:
        if not isinstance(raw_labels, list):
            return [], []
        boxes: list[list[float]] = []
        labels: list[int] = []
        for raw_label in raw_labels:
            if not isinstance(raw_label, dict):
                continue
            category = raw_label.get("category")
            if category not in self.class_to_index:
                continue
            box2d = raw_label.get("box2d")
            if not isinstance(box2d, dict):
                continue
            try:
                x0 = float(box2d["x1"])
                y0 = float(box2d["y1"])
                x1 = float(box2d["x2"])
                y1 = float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            labels.append(self.class_to_index[str(category)])
        return boxes, labels

    def _sample_from_aggregate_entry(self, entry: object, image_dir: Path) -> dict[str, object] | None:
        if not isinstance(entry, dict):
            return None
        image_name = entry.get("name")
        if not isinstance(image_name, str):
            return None
        image_path = image_dir / image_name
        if not image_path.exists():
            return None
        boxes, labels = self._parse_vehicle_annotations(entry.get("labels"))
        return {"image_path": image_path, "boxes": boxes, "labels": labels}

    def _sample_from_frame_file(self, annotation_path: Path, image_dir: Path) -> dict[str, object] | None:
        entry = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            return None
        image_name = entry.get("name")
        if not isinstance(image_name, str):
            return None
        image_path = image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            return None
        frames = entry.get("frames")
        if not isinstance(frames, list) or not frames:
            return {"image_path": image_path, "boxes": [], "labels": []}
        frame = frames[0]
        if not isinstance(frame, dict):
            return {"image_path": image_path, "boxes": [], "labels": []}
        boxes, labels = self._parse_vehicle_annotations(frame.get("objects"))
        return {"image_path": image_path, "boxes": boxes, "labels": labels}


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
