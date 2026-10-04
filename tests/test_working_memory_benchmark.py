from __future__ import annotations

from pathlib import Path

from neural_observability.working_memory.benchmark import MODELS, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_defines_the_four_models_and_the_ridge_grid() -> None:
    config, root = load_config(ROOT / "configs" / "working_memory_benchmark.json")
    assert root == ROOT
    assert tuple(config["models"]["names"]) == MODELS
    ridge = config["models"]["lagged_linear_ridge"]
    assert len(ridge["lag_tap_sets_samples_at_200_hz"]) * len(ridge["ridge_alpha_multipliers"]) == 42
    assert config["partitions"]["outer_hyperparameter_split"] == "leave_one_subject_out"
    assert config["partitions"]["test_subject_information_may_enter_hyperparameter_selection"] is False
    assert (ROOT / config["generative_config"]).is_file()
