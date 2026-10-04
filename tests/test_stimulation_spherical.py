import json
from pathlib import Path

import numpy as np
import pytest

from neural_observability.stimulation.spherical import (
    hierarchical_label_permutation_test,
    infer_bipolar_orientation,
    load_config,
    parse_stimulation_description,
    score_run_topographies,
    spatial_squared_cosine,
    subject_source_effect,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stimulation_spherical.json"


def test_configuration_loads_with_the_analysis_windows():
    config, root = load_config(CONFIG)
    assert root == ROOT
    assert config["windows_seconds"]["primary_early"] == [0.005, 0.02]
    assert config["head_model"]["nominal_skull_conductivity_s_per_m"] == 0.0042


def test_parse_stimulation_description_with_apostrophe():
    parsed = parse_stimulation_description(
        "Stimulation of R'6-7 5ma 0.5ms 0.5hz parallel wh_wh "
        "(channel, intensity, duration, frequency, angle, grey/white matter)"
    )
    assert parsed == {
        "source_label": "R'6-7",
        "intensity_ma": 5.0,
        "pulse_width_ms": 0.5,
    }


def test_parse_stimulation_description_accepts_released_msma_typo():
    parsed = parse_stimulation_description(
        "Stimulation of F'6-7 5msma 0.5ms 0.5hz perpend gr_gr "
        "(channel, intensity, duration, frequency, angle, grey/white matter)"
    )
    assert parsed["source_label"] == "F'6-7"
    assert parsed["intensity_ma"] == 5.0


def test_orientation_prefers_central_difference_and_is_unit_length():
    positions = {
        "R'5-6": np.array([0.0, 0.0, 0.0]),
        "R'6-7": np.array([1.0, 0.0, 0.0]),
        "R'7-8": np.array([2.0, 0.0, 0.0]),
    }
    orientation, method = infer_bipolar_orientation("R'6-7", positions)
    np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0])
    assert method == "central_adjacent_pairs"


def test_spatial_squared_cosine_is_reference_and_sign_invariant():
    prediction = np.array([1.0, 2.0, 4.0, -1.0])
    assert spatial_squared_cosine(prediction + 9.0, -3.0 * prediction + 2.0) == pytest.approx(1.0)


def test_subject_effect_and_permutation_detect_correct_sources():
    scores = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.1],
            [0.1, 0.2, 0.95],
            [0.85, 0.2, 0.1],
        ]
    )
    labels = np.array([0, 1, 2, 0])
    effect, run_effects = subject_source_effect(scores, labels)
    assert effect > 0.6
    assert np.all(run_effects > 0)
    result = hierarchical_label_permutation_test(
        [scores] * 8, [labels] * 8, permutations=999, seed=7
    )
    assert result["observed_mean_subject_effect"] == pytest.approx(effect)
    assert result["one_sided_p_value"] < 0.05


def test_run_scoring_respects_good_channels_and_half_open_window(tmp_path):
    # The last channel is deliberately adversarial but marked bad.
    data = np.zeros((2, 4, 10), dtype=float)
    data[:, :, 5] = np.array([1.0, -1.0, 0.5, 1e6])
    path = tmp_path / "epochs.npy"
    np.save(path, data)
    run = {
        "subject": "sub-test",
        "run": "01",
        "array_path": path,
        "channel_names": ["a", "b", "c", "bad"],
        "good_channels": ["a", "b", "c"],
        "sample_count": 10,
        "zero_time_seconds": 0.0,
    }
    predictions = {
        "right": np.array([1.0, -1.0, 0.5, -2.0]),
        "wrong": np.array([1.0, 1.0, -2.0, 3.0]),
    }
    scores = score_run_topographies(
        run, predictions, ["a", "b", "c", "bad"], [0.005, 0.006], 1000.0
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]


def test_config_rejects_non_increasing_radii(tmp_path):
    source = json.loads(CONFIG.read_text())
    source["head_model"]["relative_radii"] = [0.9, 0.9, 0.97, 1.0]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="increasing"):
        load_config(path)
