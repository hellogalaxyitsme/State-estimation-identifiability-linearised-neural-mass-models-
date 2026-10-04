import json
from pathlib import Path

import numpy as np
import pytest
import nibabel as nib

from neural_observability.stimulation.cohort import (
    baseline_standardized_trial_mean,
    bipolar_shaft,
    conductivity_sensitivity_subjects,
    dynamic_proxy_trajectory,
    fixed_ridge_fit_predict,
    holm_adjust,
    leave_one_subject_out_ridge,
    load_config,
    load_eligible_runs,
    load_visual_qc_passes,
    mean_spatial_squared_cosine,
    one_sided_sign_permutation_test,
    source_triangle_vertices,
    subject_mean_losses,
    whitened_scalp_detectability_target,
)
from neural_observability.head_models.fem import (
    electrode_projection_quality,
    oriented_source_leadfield,
    reconstruct_average_referenced_eeg_forward,
    summarize_solver_log_records,
    tetrahedral_numerical_quality,
    validate_compact_fem_artifact,
    write_simnibs_eeg_cap,
)
from neural_observability.statistics import subject_bootstrap_mean_interval
from neural_observability.stimulation.fem_analysis import (
    _brain_distance_eligible_sources,
    geometry_permutation_scores,
)
from neural_observability.head_models.fem_qc import (
    make_qc_review_index,
    make_segmentation_overlay,
    write_qc_cohort_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stimulation_fem.json"


def test_configuration_loads_with_three_tests_and_three_conductivities():
    config, root = load_config(CONFIG)
    assert root == ROOT
    assert len(config["inference"]["tests"]) == 3
    assert config["head_model"]["nominal_skull_conductivity_s_per_m"] in config["head_model"][
        "skull_conductivities_s_per_m"
    ]


def test_bipolar_shaft_and_stimulated_shaft_exclusion():
    assert bipolar_shaft("R'6-7") == "R'"
    assert bipolar_shaft("ABC12-13") == "ABC"
    assert bipolar_shaft("R'6-7") != bipolar_shaft("R6-7")
    with pytest.raises(ValueError, match="adjacent"):
        bipolar_shaft("R'6-8")


def test_sensitivity_selection_is_deterministic_and_order_independent():
    subjects = ["sub-04", "sub-02", "sub-01", "sub-03"]
    selected = conductivity_sensitivity_subjects(subjects, seed=17, count=2)
    assert selected == conductivity_sensitivity_subjects(reversed(subjects), 17, 2)
    assert len(selected) == 2


def test_source_triangle_has_exact_first_vertex_and_nonzero_area():
    point = np.array([12.0, -4.0, 22.5])
    vertices, faces = source_triangle_vertices(point, 0.05)
    np.testing.assert_array_equal(vertices[0], point)
    np.testing.assert_array_equal(faces, [[0, 1, 2]])
    area_vector = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    assert np.linalg.norm(area_vector) == pytest.approx(0.05**2)


def test_baseline_standardization_uses_only_baseline_statistics():
    rng = np.random.default_rng(3)
    times = np.arange(-0.3, 0.201, 0.001)
    trials = rng.normal(loc=2.0, scale=4.0, size=(12, 3, len(times)))
    response = (times >= 0.02) & (times < 0.1)
    trials[:, 0, response] += 10.0
    trajectory, means, scales = baseline_standardized_trial_mean(
        trials, times, [-0.25, -0.05], scale_floor=1e-8
    )
    baseline = (times >= -0.25) & (times < -0.05)
    assert abs(float(np.mean(trajectory[:, baseline]))) < 0.08
    assert float(np.mean(trajectory[0, response])) > 2.0
    assert np.all(scales > 3.5)
    assert np.all(np.isfinite(means))


def test_dynamic_projection_and_squared_cosine_are_reference_scale_sign_invariant():
    lead = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    state = np.array([[1.0, 2.0], [-2.0, 0.5]])
    prediction = dynamic_proxy_trajectory(lead, state)
    observation = -3.0 * prediction + 8.0
    assert mean_spatial_squared_cosine(prediction, observation) == pytest.approx(1.0)


def test_whitened_target_is_common_sensor_scale_invariant():
    rng = np.random.default_rng(9)
    times = np.arange(-0.3, 0.201, 0.001)
    trials = rng.normal(scale=2.0, size=(20, 6, len(times)))
    response = (times >= 0.02) & (times < 0.1)
    spatial = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    trials[:, :, response] += spatial[None, :, None]
    target, details = whitened_scalp_detectability_target(
        trials, times, [-0.25, -0.05], [0.02, 0.1], 0.1
    )
    scaled, _ = whitened_scalp_detectability_target(
        -7.0 * trials, times, [-0.25, -0.05], [0.02, 0.1], 0.1
    )
    assert target == pytest.approx(scaled, rel=1e-12)
    assert details["independent_channel_count"] == 5
    assert target > 0


def test_fixed_ridge_uses_training_scaling_and_loso_covers_every_run():
    features = np.array(
        [[0.0], [0.2], [1.0], [1.2], [2.0], [2.2], [3.0], [3.2]]
    )
    outcomes = 1.5 + 2.0 * features[:, 0]
    subjects = ["a", "a", "b", "b", "c", "c", "d", "d"]
    predictions = leave_one_subject_out_ridge(features, outcomes, subjects, alpha=0.01)
    assert predictions.shape == outcomes.shape
    assert np.mean((predictions - outcomes) ** 2) < 0.1
    losses = subject_mean_losses(outcomes, predictions, subjects)
    assert set(losses) == set(subjects)
    # A constant training feature remains finite and the intercept is unpenalized.
    constant = fixed_ridge_fit_predict(
        np.ones((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]), np.ones((2, 1)), 1.0
    )
    np.testing.assert_allclose(constant, [2.5, 2.5])


def test_sign_permutation_and_holm_detect_ordered_effects():
    differences = np.linspace(0.2, 1.0, 12)
    result = one_sided_sign_permutation_test(differences, permutations=9999, seed=4)
    assert result["one_sided_p_value"] < 0.01
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_simnibs_cap_marks_exactly_one_declared_reference(tmp_path):
    path = write_simnibs_eeg_cap(
        tmp_path / "cap.csv",
        ["A", "B", "C"],
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
        "B",
    )
    lines = path.read_text().splitlines()
    assert sum(line.startswith("ReferenceElectrode,") for line in lines) == 1
    assert lines[1].endswith(",B")


def test_reciprocal_reconstruction_restores_reference_and_average_reference():
    reciprocal = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    forward = reconstruct_average_referenced_eeg_forward(
        reciprocal, ["A", "REF", "B"], "REF"
    )
    assert forward.shape == (3, 4, 3)
    np.testing.assert_allclose(forward.sum(axis=0), 0.0, atol=1e-12)
    # Before average projection the reconstructed reference row was exactly zero.
    expected = -np.insert(reciprocal, 1, np.zeros((1, 4, 3)), axis=0)
    expected -= expected.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(forward, expected)


def _synthetic_solver_log_records():
    system = np.array([[2.0, -0.5], [-0.5, 1.5]])
    records = []
    for index in range(2):
        rhs = np.zeros(2)
        rhs[index] = 1.0
        solution = np.linalg.solve(system, rhs)
        records.append(
            {
                "simulation_index": index,
                "electrode_union_dof_indices": [0, 1],
                "rhs_nonzero_dof_indices": [index],
                "rhs_nonzero_values": [1.0],
                "solution_at_electrode_union": solution.tolist(),
                "ksp_converged_reason": 2,
                "ksp_iteration_count": 7,
                "ksp_reported_residual_norm": 1e-12,
                "unpreconditioned_relative_residual": 1e-12,
            }
        )
    return records


def test_solver_log_reconstructs_all_rhs_discrete_reciprocity():
    config, _ = load_config(CONFIG)
    summary, arrays = summarize_solver_log_records(
        _synthetic_solver_log_records(), 2, config["fem_validation"]
    )
    assert summary["solver_convergence_pass"] is True
    assert summary["discrete_reciprocity_pass"] is True
    np.testing.assert_allclose(
        arrays["reciprocity_matrix"], arrays["reciprocity_matrix"].T, atol=1e-14
    )


def test_solver_log_rejects_an_asymmetric_solution_matrix():
    config, _ = load_config(CONFIG)
    records = _synthetic_solver_log_records()
    records[1]["solution_at_electrode_union"][0] += 0.1
    with pytest.raises(RuntimeError, match="reciprocity"):
        summarize_solver_log_records(records, 2, config["fem_validation"])


def test_compact_fem_validation_recomputes_reference_and_geometry_checks(tmp_path):
    config, _ = load_config(CONFIG)
    path = tmp_path / "leadfield.npz"
    np.savez_compressed(
        path,
        oriented_lead_field=np.array([[1.0, 2.0], [-0.5, -1.0], [-0.5, -1.0]]),
        eeg_channel_names=np.array(["e1", "e2", "e3"]),
        source_labels=np.array(["A1-2", "B1-2"]),
        source_positions_mm=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        source_orientations=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        source_depth_mm=np.array([10.0, 12.0]),
    )
    result = validate_compact_fem_artifact(path, config["fem_validation"])
    assert result["shape"] == [3, 2]
    assert result["average_reference_max_column_sum_relative"] == 0.0


def test_oriented_forward_selects_only_exact_first_triangle_vertices():
    forward = np.zeros((2, 6, 3), dtype=float)
    forward[:, 0, 0] = [1.0, 2.0]
    forward[:, 3, 1] = [3.0, 4.0]
    # Large carrier-vertex values must not leak into the selected midpoint.
    forward[:, [1, 2, 4, 5], :] = 1e6
    oriented = oriented_source_leadfield(
        forward, [0, 3], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    np.testing.assert_allclose(oriented, [[1.0, 3.0], [2.0, 4.0]])


def test_tetrahedral_quality_has_known_regular_tetrahedron_answer():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
        ]
    )
    quality = tetrahedral_numerical_quality(nodes, np.array([[0, 1, 2, 3]]))
    assert quality["nonfinite_metric_count"] == 0
    assert quality["mean_ratio"]["minimum"] == pytest.approx(1.0)
    assert quality["edge_aspect_ratio"]["maximum"] == pytest.approx(1.0)
    assert quality["minimum_dihedral_angle_degrees"]["minimum"] == pytest.approx(
        70.528779, rel=1e-6
    )
    assert quality["maximum_dihedral_angle_degrees"]["maximum"] == pytest.approx(
        70.528779, rel=1e-6
    )
    assert quality["regular_reference_jacobian_condition_number"][
        "maximum"
    ] == pytest.approx(1.0)


def test_tetrahedral_quality_detects_a_sliver_and_is_permutation_invariant():
    regular = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
        ]
    )
    sliver = regular.copy()
    sliver[3, 2] = 1e-3
    first = tetrahedral_numerical_quality(sliver, np.array([[0, 1, 2, 3]]))
    permuted = tetrahedral_numerical_quality(sliver, np.array([[2, 0, 3, 1]]))
    assert first["mean_ratio"]["minimum"] < 0.02
    assert first["minimum_dihedral_angle_degrees"]["minimum"] < 1.0
    assert first["regular_reference_jacobian_condition_number"]["maximum"] > 100
    assert permuted["mean_ratio"]["minimum"] == pytest.approx(
        first["mean_ratio"]["minimum"]
    )
    assert permuted["regular_reference_jacobian_condition_number"][
        "maximum"
    ] == pytest.approx(
        first["regular_reference_jacobian_condition_number"]["maximum"]
    )


def test_electrode_projection_quality_separates_radial_scale_from_topology():
    raw = 0.8 * np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projected = raw / 0.8
    scalp = np.vstack([projected, -projected, np.zeros((1, 3))])
    landmark_names = ["LPA", "NAS", "RPA"]
    landmarks = {name: raw[index] for index, name in enumerate(landmark_names)}
    projected_landmarks = {
        name: projected[index] for index, name in enumerate(landmark_names)
    }
    quality = electrode_projection_quality(
        [f"E{index}" for index in range(len(raw))],
        raw,
        projected,
        scalp,
        landmarks,
        projected_landmarks,
        nearest_neighbour_count=2,
    )
    assert quality["projection_distance_mm"]["median"] == pytest.approx(0.2)
    assert quality["outward_projection_fraction"] == pytest.approx(1.0)
    assert quality["radial_alignment_cosine"]["median"] == pytest.approx(1.0)
    assert quality["radial_alignment_absolute_cosine"]["median"] == pytest.approx(
        1.0
    )
    assert quality["angular_displacement_degrees"]["p95"] == pytest.approx(0.0)
    assert quality["neighbour_overlap_fraction"]["mean"] == pytest.approx(1.0)
    assert quality["radial_scale_ratio_raw_over_projected"][
        "median"
    ] == pytest.approx(0.8)


def test_geometry_permutation_control_separates_correct_mapping():
    rng = np.random.default_rng(11)
    lead = rng.normal(size=(8, 5))
    state = rng.normal(size=(5, 20))
    observation = lead @ state
    observed = mean_spatial_squared_cosine(observation, observation)
    null = geometry_permutation_scores(
        lead, state, observation, permutations=99, seed=8, batch_size=16
    )
    assert observed == pytest.approx(1.0)
    assert np.quantile(null, 0.95) < observed


def test_qc_and_eligibility_ledgers_never_treat_missing_as_pass(tmp_path):
    eligibility = tmp_path / "eligibility.csv"
    eligibility.write_text(
        "subject,run,pre_charm_eligible\nsub-01,01,True\nsub-01,02,False\n"
    )
    assert load_eligible_runs(eligibility) == {"sub-01": ["01"]}
    assert load_visual_qc_passes(tmp_path / "missing.csv") == set()
    visual = tmp_path / "visual.csv"
    visual.write_text(
        "subject,visual_qc_status\nsub-01,pending\nsub-02,pass\nsub-03,fail\n"
    )
    assert load_visual_qc_passes(visual) == {"sub-02"}


def test_analysis_uses_the_mri_source_distance_criterion(tmp_path):
    qc_dir = tmp_path / "headmodel_qc"
    qc_dir.mkdir(parents=True)
    (qc_dir / "sub-01_automated_qc.json").write_text(
        json.dumps(
            {
                "automated_qc_pass": True,
                "source_distance_mm_by_label": {
                    "A1-2": 0.0,
                    "B1-2": 5.0,
                    "C1-2": 5.0001,
                },
            }
        )
    )
    config = {
        "output_dir": str(tmp_path),
        "eligibility": {"maximum_source_to_brain_distance_mm": 5.0},
    }
    assert _brain_distance_eligible_sources(config, "sub-01") == {
        "A1-2",
        "B1-2",
    }


def test_subject_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.arange(1.0, 9.0)
    first = subject_bootstrap_mean_interval(values, 5000, 13)
    second = subject_bootstrap_mean_interval(values, 5000, 13)
    assert first == second
    assert first[0] < values.mean() < first[1]


def test_static_segmentation_qc_overlay_is_created(tmp_path):
    grid = np.indices((20, 22, 24))
    radius = np.sqrt(
        ((grid[0] - 10) / 7) ** 2
        + ((grid[1] - 11) / 8) ** 2
        + ((grid[2] - 12) / 9) ** 2
    )
    anatomy = np.exp(-(radius**2))
    labels = np.zeros(anatomy.shape, dtype=np.int16)
    labels[radius < 0.9] = 5
    labels[radius < 0.75] = 4
    labels[radius < 0.6] = 3
    labels[radius < 0.45] = 2
    labels[radius < 0.3] = 1
    t1 = tmp_path / "t1.nii.gz"
    tissue = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(anatomy, np.eye(4)), t1)
    # SimNIBS 4.6 emits final_tissues as a singleton 4-D NIfTI.
    nib.save(nib.Nifti1Image(labels[..., None], np.eye(4)), tissue)
    output = make_segmentation_overlay(t1, tissue, tmp_path / "qc.png", "sub-test")
    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_visual_qc_review_index_lists_records_without_deciding(tmp_path):
    (tmp_path / "sub-test_automated_qc.json").write_text(
        json.dumps({"subject": "sub-test", "automated_qc_pass": True})
    )
    index = make_qc_review_index(tmp_path)
    contents = index.read_text()
    assert "sub-test" in contents
    assert "sub-test_segmentation_overlay.png" in contents
    assert not (tmp_path / "visual_qc.csv").exists()


def test_qc_cohort_summary_counts_failed_checks(tmp_path):
    summary = {
        "subject": "sub-test",
        "automated_qc_pass": False,
        "checks": {"mesh_pass": True, "collision_pass": False},
        "tetrahedron_count": 100,
        "invalid_tetrahedron_count": 0,
        "tetrahedral_numerical_quality": {
            "mean_ratio": {"p001": 0.2},
            "fraction_mean_ratio_below_0_01": 0.0,
            "minimum_dihedral_angle_degrees": {"p001": 8.0},
            "maximum_dihedral_angle_degrees": {"p999": 170.0},
            "edge_aspect_ratio": {"p999": 4.0},
            "regular_reference_jacobian_condition_number": {"p999": 12.0},
        },
        "electrode_projection_qc": {
            "projection_distance_mm": {"median": 9.0, "p95": 14.0, "maximum": 18.0},
            "outward_projection_fraction": 0.0,
            "radially_aligned_projection_fraction": 0.98,
            "radial_alignment_absolute_cosine": {"median": 0.95},
            "angular_displacement_degrees": {"p95": 4.0},
            "neighbour_overlap_fraction": {"mean": 0.9},
            "minimum_projected_interelectrode_distance_mm": 1.5,
            "maximum_fiducial_to_scalp_distance_mm": 7.0,
            "centroid_projection_displacement_mm": 4.0,
            "radial_scale_ratio_raw_over_projected": {"median": 1.1},
        },
        "source_to_brain_conservative_node_distance_mm": {
            "median": 0.0,
            "p95": 1.0,
            "maximum": 3.0,
        },
    }
    (tmp_path / "sub-test_automated_qc.json").write_text(json.dumps(summary))
    csv_path, json_path = write_qc_cohort_summary(tmp_path)
    result = json.loads(json_path.read_text())
    assert result["automated_fail_count"] == 1
    assert result["failure_count_by_check"] == {"collision_pass": 1}
    assert csv_path.is_file()


