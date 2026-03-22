from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image as PILImage

from easy_rtdetr.datasets import BDD100KVehicleDataset


def _write_bdd_sample(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    image = PILImage.new("RGB", (64, 32), color=(10, 20, 30))
    image.save(root / "images" / "train" / "sample.jpg")
    labels = [
        {
            "name": "sample.jpg",
            "labels": [
                {"category": "car", "box2d": {"x1": 4, "y1": 5, "x2": 20, "y2": 16}},
                {"category": "truck", "box2d": {"x1": 22, "y1": 4, "x2": 40, "y2": 18}},
                {"category": "person", "box2d": {"x1": 1, "y1": 1, "x2": 3, "y2": 10}},
            ],
        }
    ]
    (root / "labels" / "det_train_vehicle3.json").write_text(json.dumps(labels), encoding="utf-8")


def test_bdd100k_vehicle_dataset_filters_and_maps_classes(tmp_path: Path):
    _write_bdd_sample(tmp_path)
    dataset = BDD100KVehicleDataset(tmp_path, image_size=64, split="train")
    image, target = dataset[0]

    assert tuple(image.shape) == (3, 64, 64)
    assert target["labels"].tolist() == [0, 1]
    assert target["boxes"].shape == (2, 4)
    assert torch.all(target["boxes"] >= 0.0)
    assert torch.all(target["boxes"] <= 1.0)


def test_bdd100k_vehicle_dataset_positive_only_filters_empty_entries(tmp_path: Path):
    (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels").mkdir(parents=True, exist_ok=True)
    image = PILImage.new("RGB", (32, 32), color=(0, 0, 0))
    image.save(tmp_path / "images" / "val" / "empty.jpg")
    labels = [{"name": "empty.jpg", "labels": [{"category": "person", "box2d": {"x1": 1, "y1": 1, "x2": 4, "y2": 6}}]}]
    (tmp_path / "labels" / "det_val_vehicle3.json").write_text(json.dumps(labels), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        BDD100KVehicleDataset(tmp_path, image_size=32, split="val", positive_only=True)

    dataset = BDD100KVehicleDataset(tmp_path, image_size=32, split="val", positive_only=False)
    _, target = dataset[0]
    assert target["labels"].numel() == 0
    assert target["boxes"].shape == (0, 4)
