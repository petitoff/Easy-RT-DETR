from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from easy_rtdetr.cli import build_model
from easy_rtdetr.configuration import load_experiment_config
from easy_rtdetr.data import build_dataloaders, build_dataset_bundle
from easy_rtdetr.engine import Solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal Easy-RT-DETR training entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to copy the best checkpoint after training. Useful for remote runners.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    config.data["seed"] = int(config.runtime.get("seed", 0))
    bundle = build_dataset_bundle(config.data)
    train_loader, eval_loader = build_dataloaders(bundle, config.runtime, config.solver)
    solver = Solver(config, build_model(config), train_loader, eval_loader)
    summary = solver.fit()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        best_path = Path(summary["run_dir"]) / "checkpoints" / "best.pt"
        fallback_path = Path(summary["run_dir"]) / "checkpoints" / "last.pt"
        source = best_path if best_path.exists() else fallback_path
        shutil.copy2(source, output_path)
        print(f"checkpoint_saved={output_path}")


if __name__ == "__main__":
    main()
