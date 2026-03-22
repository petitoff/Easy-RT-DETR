from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..configuration import ExperimentConfig
from ..model import RTDETRv3
from ..optim import ModelEMA, build_optimizer, build_scheduler
from .checkpoints import save_checkpoint, save_json
from .evaluator import evaluate_detection_model
from .logging import setup_logger
from .profiler import EpochProfiler


@dataclass(slots=True)
class SolverArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    metrics_path: Path
    resolved_config_path: Path


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


class Solver:
    def __init__(
        self,
        config: ExperimentConfig,
        model: RTDETRv3,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ) -> None:
        self.config = config
        self.device = resolve_device(str(config.runtime.get("device", "auto")))
        if self.device.type == "cuda" and bool(config.runtime.get("cudnn_benchmark", False)):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(config.runtime.get("matmul_precision", "high")))
        seed = int(config.runtime.get("seed", 0))
        torch.manual_seed(seed)
        random.seed(seed)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = build_optimizer(self.model, config.solver)
        self.scheduler = build_scheduler(self.optimizer, config.solver)
        self.use_amp = bool(config.runtime.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = ModelEMA(self.model, decay=float(config.solver.get("ema_decay", 0.999))) if bool(config.solver.get("use_ema", False)) else None
        self.artifacts = self._create_run_dir()
        self.logger = setup_logger(self.artifacts.run_dir)
        self._write_resolved_config()

    def _create_run_dir(self) -> SolverArtifacts:
        root = Path(str(self.config.runtime.get("output_dir", "runs")))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / f"{timestamp}-{self.config.project_name}"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return SolverArtifacts(
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            metrics_path=run_dir / "metrics.json",
            resolved_config_path=run_dir / "config_resolved.yaml",
        )

    def _write_resolved_config(self) -> None:
        import yaml

        self.artifacts.resolved_config_path.write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

    def fit(self) -> dict[str, Any]:
        best_metric_name = str(self.config.solver.get("best_metric", "mAP@0.50:0.95"))
        maximize_best_metric = bool(self.config.solver.get("maximize_best_metric", True))
        best_metric = -math.inf if maximize_best_metric else math.inf
        history: list[dict[str, float]] = []
        started = time.time()

        self.logger.info("run_dir=%s", self.artifacts.run_dir)
        self.logger.info("device=%s amp=%s", self.device, self.use_amp)
        self.logger.info("train_batches=%s eval_batches=%s", len(self.train_loader), len(self.eval_loader))

        epochs = int(self.config.solver["epochs"])
        for epoch in range(1, epochs + 1):
            epoch_loss, train_metrics = self._train_one_epoch(epoch)
            epoch_record: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss, **train_metrics}
            if self.scheduler is not None:
                self.scheduler.step()
                epoch_record["lr"] = float(self.optimizer.param_groups[-1]["lr"])
            else:
                epoch_record["lr"] = float(self.optimizer.param_groups[-1]["lr"])

            if epoch % int(self.config.solver.get("eval_interval", 1)) == 0:
                eval_model = self.ema.module if self.ema is not None else self.model
                eval_metrics = evaluate_detection_model(
                    eval_model,
                    self.eval_loader,
                    self.device,
                    num_classes=self.model.config.num_classes,
                    evaluation_cfg=self.config.evaluation,
                )
                epoch_record.update(eval_metrics)
                self.logger.info("epoch=%s eval=%s", epoch, json.dumps(eval_metrics, sort_keys=True))
                current = float(eval_metrics[best_metric_name])
                is_better = current > best_metric if maximize_best_metric else current < best_metric
                if is_better:
                    best_metric = current
                    save_checkpoint(
                        self.artifacts.checkpoints_dir / "best.pt",
                        self.model,
                        self.config.to_dict(),
                        epoch,
                        eval_metrics,
                        ema_state_dict=(self.ema.module.state_dict() if self.ema is not None else None),
                    )
                    self.logger.info("best_checkpoint=%s metric=%s value=%.4f", self.artifacts.checkpoints_dir / "best.pt", best_metric_name, current)

            if epoch % int(self.config.solver.get("save_interval", 1)) == 0:
                save_checkpoint(
                    self.artifacts.checkpoints_dir / "last.pt",
                    self.model,
                    self.config.to_dict(),
                    epoch,
                    epoch_record,
                    ema_state_dict=(self.ema.module.state_dict() if self.ema is not None else None),
                )

            history.append(epoch_record)
            self.logger.info("epoch=%s train=%s", epoch, json.dumps(epoch_record, sort_keys=True))

        summary = {
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "history": history,
            "training_time_sec": time.time() - started,
            "run_dir": str(self.artifacts.run_dir),
        }
        save_json(self.artifacts.metrics_path, summary)
        return summary

    def _train_one_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train()
        running_loss = 0.0
        steps = 0
        profiler = EpochProfiler()
        log_interval = int(self.config.logging.get("log_interval", 10))
        grad_clip = float(self.config.solver.get("grad_clip_norm", 1.0))
        data_timer = time.perf_counter()

        for step, (images, targets) in enumerate(self.train_loader, start=1):
            data_time = time.perf_counter() - data_timer
            iter_start = time.perf_counter()
            images = images.to(self.device)
            targets = [
                {
                    "labels": target["labels"].to(self.device),
                    "boxes": target["boxes"].to(self.device),
                }
                for target in targets
            ]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                losses = self.model(images, targets, epoch=epoch)
                loss = losses["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)

            iter_time = time.perf_counter() - iter_start
            profiler.update(data_time, iter_time, images.size(0), self.device)
            running_loss += float(loss.item())
            steps += 1

            if step % log_interval == 0 or step == len(self.train_loader):
                self.logger.info(
                    "epoch=%s step=%s/%s loss=%.4f data_time=%.4f iter_time=%.4f",
                    epoch,
                    step,
                    len(self.train_loader),
                    loss.item(),
                    data_time,
                    iter_time,
                )
            data_timer = time.perf_counter()

        metrics = profiler.summary() if bool(self.config.logging.get("profile", True)) else {}
        return running_loss / max(steps, 1), metrics
