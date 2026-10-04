from __future__ import annotations

from pathlib import Path

from neural_observability.stimulation.matched_sources import TESTS, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_defines_the_two_matched_contrasts() -> None:
    config, root = load_config(ROOT / "configs" / "stimulation_matched_sources.json")
    assert root == ROOT
    assert tuple(config["inference"]["tests"]) == TESTS
    assert config["feature_matched_control"]["match_count"] == 10
    assert config["spatial_neighbour_control"]["match_count"] == 1
    assert config["endpoint"]["window_seconds_half_open"] == [0.005, 0.02]
    assert config["inference"]["permutations"] == 50000
    assert config["inference"]["bootstrap_resamples"] == 50000
    assert (ROOT / config["fem_config"]).is_file()
