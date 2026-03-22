from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpochProfiler:
    data_time_sum: float = 0.0
    iter_time_sum: float = 0.0
    samples: int = 0
    steps: int = 0
    max_memory_mb: float = 0.0

    def update(self, data_time: float, iter_time: float, batch_size: int, device: torch.device) -> None:
        self.data_time_sum += data_time
        self.iter_time_sum += iter_time
        self.samples += batch_size
        self.steps += 1
        if device.type == "cuda":
            self.max_memory_mb = max(self.max_memory_mb, torch.cuda.max_memory_allocated(device) / (1024**2))

    def summary(self) -> dict[str, float]:
        avg_iter_time = self.iter_time_sum / max(self.steps, 1)
        avg_data_time = self.data_time_sum / max(self.steps, 1)
        throughput = self.samples / max(self.iter_time_sum, 1e-8)
        return {
            "avg_iter_time": avg_iter_time,
            "avg_data_time": avg_data_time,
            "throughput": throughput,
            "max_memory_mb": self.max_memory_mb,
        }
