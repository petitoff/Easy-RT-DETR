from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_raw_labels(label_path: Path, keep_classes: set[str]) -> list[dict]:
    entries = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected BDD100K label format in {label_path}")
    filtered_entries: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_labels = entry.get("labels")
        if not isinstance(raw_labels, list):
            continue
        filtered_labels = []
        for raw_label in raw_labels:
            if not isinstance(raw_label, dict) or raw_label.get("category") not in keep_classes:
                continue
            box2d = raw_label.get("box2d")
            if not isinstance(box2d, dict):
                continue
            try:
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            filtered_labels.append(raw_label)
        if filtered_labels:
            new_entry = dict(entry)
            new_entry["labels"] = filtered_labels
            filtered_entries.append(new_entry)
    return filtered_entries


def parse_frame_labels(label_dir: Path, keep_classes: set[str]) -> list[dict]:
    filtered_entries: list[dict] = []
    for annotation_path in sorted(label_dir.glob("*.json")):
        entry = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            continue
        image_name = entry.get("name")
        frames = entry.get("frames")
        if not isinstance(image_name, str) or not isinstance(frames, list) or not frames:
            continue
        frame = frames[0]
        if not isinstance(frame, dict):
            continue
        raw_objects = frame.get("objects")
        if not isinstance(raw_objects, list):
            continue
        filtered_labels = []
        for raw_label in raw_objects:
            if not isinstance(raw_label, dict) or raw_label.get("category") not in keep_classes:
                continue
            box2d = raw_label.get("box2d")
            if not isinstance(box2d, dict):
                continue
            try:
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            filtered_labels.append(raw_label)
        if filtered_labels:
            filtered_entries.append({"name": f"{image_name}.jpg", "labels": filtered_labels})
    return filtered_entries


def materialize_bdd_vehicle3(raw_root: Path, output_root: Path, splits: list[str], keep_classes: set[str], mode: str) -> None:
    for split in splits:
        label_candidates = [
            raw_root / "labels" / "det_20" / f"det_{split}.json",
            raw_root / "labels" / f"det_{split}.json",
        ]
        label_dir_candidates = [
            raw_root / "labels" / "100k" / split,
            raw_root / "100k" / split,
        ]
        label_path = next((candidate for candidate in label_candidates if candidate.exists()), None)
        label_dir = next((candidate for candidate in label_dir_candidates if candidate.exists()), None)
        if label_path is None and label_dir is None:
            raise FileNotFoundError(f"BDD100K labels not found for split={split}")

        image_dir_candidates = [
            raw_root / "images" / "100k" / split,
            raw_root / "images" / split,
            raw_root / "10k" / split,
        ]
        image_dir = next((candidate for candidate in image_dir_candidates if candidate.exists()), None)
        if image_dir is None:
            raise FileNotFoundError(f"BDD100K images not found for split={split}")

        entries = parse_raw_labels(label_path, keep_classes) if label_path is not None else parse_frame_labels(label_dir, keep_classes)
        output_image_dir = output_root / "images" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir = output_root / "labels"
        output_label_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        kept_entries: list[dict] = []
        for entry in entries:
            image_name = entry.get("name")
            if not isinstance(image_name, str):
                continue
            source_image = image_dir / image_name
            if not source_image.exists():
                continue
            target_image = output_image_dir / image_name
            if not target_image.exists():
                if mode == "copy":
                    shutil.copy2(source_image, target_image)
                elif mode == "hardlink":
                    target_image.hardlink_to(source_image)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                copied += 1
            kept_entries.append(entry)

        output_label_path = output_label_dir / f"det_{split}_vehicle3.json"
        output_label_path.write_text(json.dumps(kept_entries), encoding="utf-8")
        print(f"split={split} kept_images={len(kept_entries)} copied_images={copied}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal dataset preparation entrypoint.")
    parser.add_argument("--dataset", required=True, choices=["bdd_vehicle3"])
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", default="car,truck,bus")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--mode", default="copy", choices=("copy", "hardlink"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    keep_classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if args.dataset == "bdd_vehicle3":
        materialize_bdd_vehicle3(raw_root, output_root, splits, keep_classes, args.mode)
        return
    raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
