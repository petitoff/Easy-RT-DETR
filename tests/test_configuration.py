from pathlib import Path

from easy_rtdetr.configuration import apply_overrides, load_experiment_config


def test_apply_overrides_updates_nested_values():
    config = {
        "solver": {"epochs": 1},
        "runtime": {"amp": False},
    }
    updated = apply_overrides(config, ["solver.epochs=20", "runtime.amp=true", "model.backbone_name=resnet50"])
    assert updated["solver"]["epochs"] == 20
    assert updated["runtime"]["amp"] is True
    assert updated["model"]["backbone_name"] == "resnet50"


def test_load_experiment_config_builds_model_config(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
project_name: smoke
model:
  preset: rtdetrv3_r18
  num_classes: 1
  hidden_dim: 64
data:
  name: synthetic
  image_size: 64
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path, overrides=["solver.epochs=3"])
    model_config = config.build_model_config()
    assert config.project_name == "smoke"
    assert config.solver["epochs"] == 3
    assert model_config.backbone_name == "resnet18"
    assert model_config.hidden_dim == 64
