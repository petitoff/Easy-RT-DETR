from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import RTDETRv3Config


DEFAULT_EXPERIMENT: dict[str, Any] = {
    "project_name": "easy-rtdetr",
    "model": {},
    "data": {
        "name": "voc_car",
        "image_size": 256,
        "train_fraction": 0.8,
        "train_transforms": {},
        "eval_transforms": {},
    },
    "solver": {
        "epochs": 1,
        "batch_size": 2,
        "optimizer": "adamw",
        "lr": 3e-4,
        "backbone_lr": None,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 0,
        "warmup_start_factor": 0.2,
        "step_size": 10,
        "step_gamma": 0.1,
        "grad_clip_norm": 1.0,
        "use_ema": False,
        "ema_decay": 0.999,
        "eval_interval": 1,
        "save_interval": 1,
        "best_metric": "mAP@0.50:0.95",
        "maximize_best_metric": True,
    },
    "runtime": {
        "device": "auto",
        "num_workers": 0,
        "pin_memory": False,
        "amp": False,
        "seed": 0,
        "output_dir": "runs",
        "cudnn_benchmark": False,
        "matmul_precision": "high",
    },
    "logging": {
        "log_interval": 10,
        "profile": True,
    },
    "evaluation": {
        "score_threshold": 0.18,
        "nms_threshold": 0.25,
        "topk": 20,
        "match_iou": 0.5,
        "duplicate_iou": 0.4,
        "calibration_path": None,
    },
}


@dataclass(slots=True)
class ExperimentConfig:
    project_name: str
    model: dict[str, Any]
    data: dict[str, Any]
    solver: dict[str, Any]
    runtime: dict[str, Any]
    logging: dict[str, Any]
    evaluation: dict[str, Any]
    source_path: Path | None = None

    def build_model_config(self) -> RTDETRv3Config:
        values = copy.deepcopy(self.model)
        preset = values.pop("preset", None)
        if preset is not None:
            return RTDETRv3Config.preset(preset, **values)
        return RTDETRv3Config(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "model": copy.deepcopy(self.model),
            "data": copy.deepcopy(self.data),
            "solver": copy.deepcopy(self.solver),
            "runtime": copy.deepcopy(self.runtime),
            "logging": copy.deepcopy(self.logging),
            "evaluation": copy.deepcopy(self.evaluation),
        }


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}. Expected key=value.")
        key, raw_value = override.split("=", 1)
        value = parse_override(raw_value)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Invalid override key: {key}")
        cursor: dict[str, Any] = updated
        for part in parts[:-1]:
            nested = cursor.get(part)
            if nested is None:
                cursor[part] = {}
                nested = cursor[part]
            if not isinstance(nested, dict):
                raise ValueError(f"Cannot override nested key on non-mapping path: {key}")
            cursor = nested
        cursor[parts[-1]] = value
    return updated


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"Top-level config must be a mapping: {config_path}")
    merged = deep_merge(DEFAULT_EXPERIMENT, raw_data)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return ExperimentConfig(
        project_name=str(merged["project_name"]),
        model=dict(merged["model"]),
        data=dict(merged["data"]),
        solver=dict(merged["solver"]),
        runtime=dict(merged["runtime"]),
        logging=dict(merged["logging"]),
        evaluation=dict(merged["evaluation"]),
        source_path=config_path,
    )
