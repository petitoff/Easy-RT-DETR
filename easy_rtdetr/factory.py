from __future__ import annotations

from .config import RTDETRv3Config
from .model import RTDETRv3


def build_model(name: str = "rtdetrv3_r18", **overrides) -> RTDETRv3:
    config = RTDETRv3Config.preset(name, **overrides)
    return RTDETRv3(config)
