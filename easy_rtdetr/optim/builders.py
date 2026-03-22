from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR, StepLR


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not value.dtype.is_floating_point:
                value.copy_(model_value)
                continue
            value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


def build_optimizer(model: torch.nn.Module, solver_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(solver_cfg["lr"])
    backbone_lr = solver_cfg.get("backbone_lr")
    backbone_lr = lr if backbone_lr is None else float(backbone_lr)
    weight_decay = float(solver_cfg.get("weight_decay", 1e-4))

    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    param_groups = []
    if backbone_parameters:
        param_groups.append({"params": backbone_parameters, "lr": backbone_lr})
    if other_parameters:
        param_groups.append({"params": other_parameters, "lr": lr})

    optimizer_name = str(solver_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    solver_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    total_epochs = int(solver_cfg["epochs"])
    if total_epochs <= 0:
        return None
    scheduler_name = str(solver_cfg.get("lr_scheduler", "cosine")).lower()
    warmup_epochs = int(solver_cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(solver_cfg.get("warmup_start_factor", 0.2))

    if scheduler_name == "constant":
        main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)
    elif scheduler_name == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=int(solver_cfg.get("step_size", max(total_epochs // 2, 1))),
            gamma=float(solver_cfg.get("step_gamma", 0.1)),
        )
    elif scheduler_name == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1))
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    if warmup_epochs <= 0:
        return main_scheduler

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
