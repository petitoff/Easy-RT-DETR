from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..config import RTDETRv3Config
from ..model import RTDETRv3


def save_checkpoint(
    path: Path,
    model: RTDETRv3,
    config_dict: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    ema_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "config": model.config.to_dict(),
            "experiment_config": config_dict,
            "image_size": config_dict.get("data", {}).get("image_size"),
            "seed": config_dict.get("runtime", {}).get("seed"),
            "dataset": config_dict.get("data", {}).get("name"),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    use_ema: bool = True,
) -> tuple[RTDETRv3, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config")
    if config_dict is None:
        raise KeyError(f"Checkpoint missing model config: {checkpoint_path}")
    config = RTDETRv3Config(**config_dict)
    config.pretrained_backbone = False
    model = RTDETRv3(config)
    state_dict = checkpoint.get("ema_state_dict") if use_ema and checkpoint.get("ema_state_dict") is not None else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    return model, checkpoint


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
