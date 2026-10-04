"""Individual finite-element head models and lead fields with SimNIBS 4.6.

Importing this module does not require SimNIBS. Functions that construct or
read SimNIBS meshes import it lazily and therefore need a SimNIBS 4.6
installation. Array ordering is always returned explicitly with the lead field.
"""

from __future__ import annotations

import csv
import fcntl
import json
import multiprocessing
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neural_observability.stimulation.geometry import infer_bipolar_orientation
from neural_observability.stimulation.cohort import _subject_geometry_headers, source_triangle_vertices


_SOLVER_LOG_JSONL: Path | None = None
_SOLVER_LOG_UNION_NODE_NUMBERS: np.ndarray | None = None
_SOLVER_LOG_CURRENT_SIMULATION_INDEX: int | None = None
_SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX = 0
_SOLVER_LOG_THRESHOLDS: dict[str, Any] | None = None
_ORIGINAL_KSP_SOLVE_SINGLE = None
_ORIGINAL_FEM_SOLVE = None
_ORIGINAL_TDCS_LEADFIELD = None
_ORIGINAL_RUN_TDCS_LEADFIELD = None


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("Incomplete solver-log append")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _logged_ksp_solve_single(solver, b):
    result = _ORIGINAL_KSP_SOLVE_SINGLE(solver, b)
    residual = solver._b.duplicate()
    try:
        solver.A.mult(solver._x, residual)
        residual.aypx(-1.0, solver._b)
        rhs_norm = float(solver._b.norm())
        residual_norm = float(residual.norm())
    finally:
        residual.destroy()
    if not np.isfinite(rhs_norm) or rhs_norm <= 0:
        raise RuntimeError("FEM solver log encountered a zero or invalid RHS")
    relative = residual_norm / rhs_norm
    metrics = {
        "ksp_converged_reason": int(solver.ksp.getConvergedReason()),
        "ksp_iteration_count": int(solver.ksp.getIterationNumber()),
        "ksp_reported_residual_norm": float(solver.ksp.getResidualNorm()),
        "rhs_norm": rhs_norm,
        "unpreconditioned_residual_norm": residual_norm,
        "unpreconditioned_relative_residual": relative,
    }
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("FEM solver convergence metrics are nonfinite")
    thresholds = _SOLVER_LOG_THRESHOLDS or {}
    if metrics["ksp_converged_reason"] < int(
        thresholds["minimum_positive_converged_reason"]
    ):
        raise RuntimeError("PETSc did not report a converged FEM solve")
    if relative > float(thresholds["maximum_unpreconditioned_relative_residual"]):
        raise RuntimeError("FEM solve exceeded the maximum unpreconditioned residual")
    solver._last_solver_log = metrics
    return result


def _logged_fem_solve(system, b=None):
    global _SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX
    if b is None:
        raise RuntimeError("The logged reciprocal FEM solve requires an explicit RHS")
    rhs = np.asarray(b, dtype=float).copy()
    solution = _ORIGINAL_FEM_SOLVE(system, b)
    if _SOLVER_LOG_JSONL is None or _SOLVER_LOG_UNION_NODE_NUMBERS is None:
        raise RuntimeError("The FEM solver log context was not initialized")
    union_dof = np.asarray(
        system.dof_map[_SOLVER_LOG_UNION_NODE_NUMBERS], dtype=int
    ).reshape(-1)
    if len(np.unique(union_dof)) != len(union_dof):
        raise RuntimeError("Electrode-support nodes do not map to unique FEM DOFs")
    rhs_nonzero = np.flatnonzero(rhs)
    if not set(rhs_nonzero).issubset(set(union_dof.tolist())):
        raise RuntimeError("A reciprocal RHS has support outside electrode nodes")
    simulation_index = _SOLVER_LOG_CURRENT_SIMULATION_INDEX
    if simulation_index is None:
        simulation_index = _SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX
        _SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX += 1
    metrics = getattr(system._solver, "_last_solver_log", None)
    if not isinstance(metrics, dict):
        raise RuntimeError("The FEM solution is missing KSP convergence evidence")
    electrode_potentials = np.asarray(solution, dtype=float)[union_dof]
    if not np.all(np.isfinite(electrode_potentials)):
        raise RuntimeError("The FEM electrode-support potentials are nonfinite")
    _append_json_line(
        _SOLVER_LOG_JSONL,
        {
            "simulation_index": int(simulation_index),
            "electrode_union_dof_indices": union_dof.tolist(),
            "rhs_nonzero_dof_indices": rhs_nonzero.tolist(),
            "rhs_nonzero_values": rhs[rhs_nonzero].tolist(),
            "solution_at_electrode_union": electrode_potentials.tolist(),
            **metrics,
        },
    )
    return solution


def _logged_run_tdcs_leadfield(*args, **kwargs):
    global _SOLVER_LOG_CURRENT_SIMULATION_INDEX
    _SOLVER_LOG_CURRENT_SIMULATION_INDEX = int(args[0])
    try:
        return _ORIGINAL_RUN_TDCS_LEADFIELD(*args, **kwargs)
    finally:
        _SOLVER_LOG_CURRENT_SIMULATION_INDEX = None


def _logged_tdcs_leadfield(*args, **kwargs):
    global _SOLVER_LOG_UNION_NODE_NUMBERS
    if len(args) < 3:
        raise RuntimeError("Unexpected SimNIBS tdcs_leadfield call signature")
    electrode_surface = args[2]
    _SOLVER_LOG_UNION_NODE_NUMBERS = np.unique(
        np.concatenate(
            [np.asarray(nodes, dtype=int).reshape(-1) for nodes in electrode_surface]
        )
    )
    if len(_SOLVER_LOG_UNION_NODE_NUMBERS) < 2:
        raise RuntimeError("The reciprocal FEM solve found too few electrode nodes")
    return _ORIGINAL_TDCS_LEADFIELD(*args, **kwargs)


@contextmanager
def logged_simnibs_solver(
    log_jsonl: str | Path, thresholds: dict[str, Any], cpus: int
):
    """Capture convergence and reciprocity evidence from every SimNIBS solve."""
    from simnibs.simulation import fem

    global _SOLVER_LOG_JSONL, _SOLVER_LOG_THRESHOLDS, _SOLVER_LOG_UNION_NODE_NUMBERS
    global _SOLVER_LOG_CURRENT_SIMULATION_INDEX, _SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX
    global _ORIGINAL_KSP_SOLVE_SINGLE, _ORIGINAL_FEM_SOLVE
    global _ORIGINAL_TDCS_LEADFIELD, _ORIGINAL_RUN_TDCS_LEADFIELD

    log_jsonl = Path(log_jsonl)
    if log_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite solver log: {log_jsonl}")
    start_method = multiprocessing.get_start_method(allow_none=True) or "fork"
    if cpus > 1 and start_method != "fork":
        raise RuntimeError("Parallel FEM solver loging requires the Linux fork method")
    _SOLVER_LOG_JSONL = log_jsonl
    _SOLVER_LOG_THRESHOLDS = thresholds
    _SOLVER_LOG_UNION_NODE_NUMBERS = None
    _SOLVER_LOG_CURRENT_SIMULATION_INDEX = None
    _SOLVER_LOG_SEQUENTIAL_SIMULATION_INDEX = 0
    _ORIGINAL_KSP_SOLVE_SINGLE = fem.KSPSolver._solve_single
    _ORIGINAL_FEM_SOLVE = fem.FEMSystem.solve
    _ORIGINAL_TDCS_LEADFIELD = fem.tdcs_leadfield
    _ORIGINAL_RUN_TDCS_LEADFIELD = fem._run_tdcs_leadfield
    fem.KSPSolver._solve_single = _logged_ksp_solve_single
    fem.FEMSystem.solve = _logged_fem_solve
    fem.tdcs_leadfield = _logged_tdcs_leadfield
    fem._run_tdcs_leadfield = _logged_run_tdcs_leadfield
    try:
        yield
    finally:
        fem.KSPSolver._solve_single = _ORIGINAL_KSP_SOLVE_SINGLE
        fem.FEMSystem.solve = _ORIGINAL_FEM_SOLVE
        fem.tdcs_leadfield = _ORIGINAL_TDCS_LEADFIELD
        fem._run_tdcs_leadfield = _ORIGINAL_RUN_TDCS_LEADFIELD
        _SOLVER_LOG_JSONL = None
        _SOLVER_LOG_THRESHOLDS = None
        _SOLVER_LOG_UNION_NODE_NUMBERS = None
        _SOLVER_LOG_CURRENT_SIMULATION_INDEX = None


def summarize_solver_log_records(
    records: Sequence[dict[str, Any]],
    expected_solve_count: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate all solve records and reconstruct the discrete reciprocity matrix."""
    if len(records) != expected_solve_count:
        raise RuntimeError(
            f"Expected {expected_solve_count} FEM solves, found {len(records)}"
        )
    ordered = sorted(records, key=lambda row: int(row["simulation_index"]))
    indices = np.asarray([int(row["simulation_index"]) for row in ordered], dtype=int)
    if not np.array_equal(indices, np.arange(expected_solve_count)):
        raise RuntimeError("FEM solver log has missing or duplicate simulation indices")
    union = np.asarray(ordered[0]["electrode_union_dof_indices"], dtype=int)
    if len(union) < 2 or len(np.unique(union)) != len(union):
        raise RuntimeError("Invalid electrode-union DOF list in solver log")
    if any(
        not np.array_equal(
            np.asarray(row["electrode_union_dof_indices"], dtype=int), union
        )
        for row in ordered[1:]
    ):
        raise RuntimeError("Electrode-union DOF ordering changed across FEM solves")
    union_lookup = {int(dof): index for index, dof in enumerate(union)}
    rhs = np.zeros((expected_solve_count, len(union)), dtype=float)
    potentials = np.empty_like(rhs)
    for index, row in enumerate(ordered):
        support = np.asarray(row["rhs_nonzero_dof_indices"], dtype=int)
        values = np.asarray(row["rhs_nonzero_values"], dtype=float)
        if len(support) < 1 or values.shape != support.shape:
            raise RuntimeError("Invalid sparse RHS in solver log")
        try:
            columns = np.asarray(
                [union_lookup[int(dof)] for dof in support], dtype=int
            )
        except KeyError as error:
            raise RuntimeError("Solver-log RHS is outside the electrode union") from error
        rhs[index, columns] = values
        potentials[index] = np.asarray(
            row["solution_at_electrode_union"], dtype=float
        )
    convergence_reasons = np.asarray(
        [int(row["ksp_converged_reason"]) for row in ordered], dtype=int
    )
    iterations = np.asarray(
        [int(row["ksp_iteration_count"]) for row in ordered], dtype=int
    )
    reported_residuals = np.asarray(
        [float(row["ksp_reported_residual_norm"]) for row in ordered], dtype=float
    )
    relative_residuals = np.asarray(
        [float(row["unpreconditioned_relative_residual"]) for row in ordered],
        dtype=float,
    )
    if not all(
        np.all(np.isfinite(array))
        for array in [rhs, potentials, reported_residuals, relative_residuals]
    ):
        raise RuntimeError("FEM solver log contains nonfinite values")
    if np.any(
        convergence_reasons
        < int(thresholds["minimum_positive_converged_reason"])
    ):
        raise RuntimeError("At least one FEM solve did not converge")
    maximum_residual = float(np.max(relative_residuals))
    if maximum_residual > float(
        thresholds["maximum_unpreconditioned_relative_residual"]
    ):
        raise RuntimeError("At least one FEM solve exceeds the residual threshold")
    reciprocity = rhs @ potentials.T
    asymmetry = reciprocity - reciprocity.T
    frobenius_error = float(
        np.linalg.norm(asymmetry, ord="fro")
        / max(np.linalg.norm(reciprocity, ord="fro"), np.finfo(float).tiny)
    )
    maximum_error = float(
        np.max(np.abs(asymmetry))
        / max(np.max(np.abs(np.diag(reciprocity))), np.finfo(float).tiny)
    )
    if frobenius_error > float(
        thresholds["maximum_reciprocity_relative_frobenius_error"]
    ):
        raise RuntimeError("The FEM solutions exceed the Frobenius reciprocity tolerance")
    if maximum_error > float(
        thresholds[
            "maximum_reciprocity_max_absolute_error_relative_to_diagonal"
        ]
    ):
        raise RuntimeError("The FEM solutions exceed the maximum reciprocity tolerance")
    summary = {
        "solve_count": int(expected_solve_count),
        "all_ksp_converged": True,
        "minimum_ksp_converged_reason": int(np.min(convergence_reasons)),
        "iteration_count": {
            "minimum": int(np.min(iterations)),
            "median": float(np.median(iterations)),
            "maximum": int(np.max(iterations)),
        },
        "unpreconditioned_relative_residual": {
            "median": float(np.median(relative_residuals)),
            "p95": float(np.quantile(relative_residuals, 0.95)),
            "maximum": maximum_residual,
        },
        "reciprocity_relative_frobenius_error": frobenius_error,
        "reciprocity_max_absolute_error_relative_to_diagonal": maximum_error,
        "solver_convergence_pass": True,
        "discrete_reciprocity_pass": True,
    }
    arrays = {
        "simulation_index": indices,
        "electrode_union_dof_indices": union,
        "rhs_on_electrode_union": rhs,
        "potential_at_electrode_union": potentials,
        "reciprocity_matrix": reciprocity,
        "ksp_converged_reason": convergence_reasons,
        "ksp_iteration_count": iterations,
        "ksp_reported_residual_norm": reported_residuals,
        "unpreconditioned_relative_residual": relative_residuals,
    }
    return summary, arrays


def finalize_solver_log(
    log_jsonl: str | Path,
    log_npz: str | Path,
    nonreference_electrode_names: Sequence[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    log_jsonl = Path(log_jsonl)
    log_npz = Path(log_npz)
    records = [
        json.loads(line) for line in log_jsonl.read_text().splitlines() if line
    ]
    summary, arrays = summarize_solver_log_records(
        records, len(nonreference_electrode_names), thresholds
    )
    np.savez_compressed(
        log_npz,
        **arrays,
        nonreference_electrode_names=np.asarray(
            nonreference_electrode_names, dtype=str
        ),
    )
    summary.update({"raw_log_path": str(log_jsonl), "log_npz_path": str(log_npz)})
    return summary


def validate_compact_fem_artifact(
    path: str | Path, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Independently validate a persisted oriented FEM operator."""
    path = Path(path)
    with np.load(path) as archive:
        required = {
            "oriented_lead_field",
            "eeg_channel_names",
            "source_labels",
            "source_positions_mm",
            "source_orientations",
            "source_depth_mm",
        }
        if not required <= set(archive.files):
            raise RuntimeError("Compact FEM artifact is missing required arrays")
        lead = np.asarray(archive["oriented_lead_field"], dtype=float)
        channel_names = np.asarray(archive["eeg_channel_names"], dtype=str)
        source_labels = np.asarray(archive["source_labels"], dtype=str)
        positions = np.asarray(archive["source_positions_mm"], dtype=float)
        orientations = np.asarray(archive["source_orientations"], dtype=float)
        depths = np.asarray(archive["source_depth_mm"], dtype=float)
    if lead.shape != (len(channel_names), len(source_labels)):
        raise RuntimeError("Compact FEM lead-field dimensions are inconsistent")
    if positions.shape != (len(source_labels), 3) or orientations.shape != positions.shape:
        raise RuntimeError("Compact FEM source geometry is inconsistent")
    if depths.shape != (len(source_labels),):
        raise RuntimeError("Compact FEM source-depth dimensions are inconsistent")
    if len(set(channel_names.tolist())) != len(channel_names):
        raise RuntimeError("Compact FEM channel names are not unique")
    if len(set(source_labels.tolist())) != len(source_labels):
        raise RuntimeError("Compact FEM source labels are not unique")
    if not all(
        np.all(np.isfinite(array))
        for array in [lead, positions, orientations, depths]
    ):
        raise RuntimeError("Compact FEM artifact contains nonfinite values")
    column_norms = np.linalg.norm(lead, axis=0)
    if np.any(column_norms <= 0):
        raise RuntimeError("Compact FEM artifact contains a zero source column")
    maximum_absolute = float(np.max(np.abs(lead)))
    reference_residual = float(
        np.max(np.abs(np.sum(lead, axis=0)))
        / max(maximum_absolute, np.finfo(float).tiny)
    )
    if reference_residual > float(
        thresholds["maximum_compact_average_reference_relative_residual"]
    ):
        raise RuntimeError("Compact FEM average-reference residual exceeds its tolerance")
    orientation_error = float(
        np.max(np.abs(np.linalg.norm(orientations, axis=1) - 1.0))
    )
    if orientation_error > float(
        thresholds["maximum_source_orientation_unit_norm_error"]
    ):
        raise RuntimeError("Compact FEM source orientations are not unit vectors")
    if np.any(depths <= 0):
        raise RuntimeError("Compact FEM source depths are not positive")
    return {
        "shape": [int(value) for value in lead.shape],
        "all_values_finite": True,
        "all_source_columns_nonzero": True,
        "minimum_source_column_norm": float(np.min(column_norms)),
        "maximum_source_column_norm": float(np.max(column_norms)),
        "average_reference_max_column_sum_relative": reference_residual,
        "source_orientation_unit_norm_max_error": orientation_error,
        "source_depths_positive_and_finite": True,
        "eeg_channel_names_unique": True,
        "source_labels_unique": True,
    }


def write_simnibs_eeg_cap(
    path: str | Path,
    channel_names: Sequence[str],
    positions_mm: np.ndarray,
    reference_name: str,
) -> Path:
    """Write native-MRI digitization in the SimNIBS electrode CSV format."""
    path = Path(path)
    names = list(channel_names)
    positions = np.asarray(positions_mm, dtype=float)
    if positions.shape != (len(names), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("EEG names and finite millimetre positions are incompatible")
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("EEG channel names must be unique")
    if reference_name not in names:
        raise ValueError("reference_name is absent from the montage")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for name, position in zip(names, positions):
            kind = "ReferenceElectrode" if name == reference_name else "Electrode"
            writer.writerow([kind, *[f"{value:.12g}" for value in position], name])
    return path


def write_simnibs_source_mesh(
    path: str | Path,
    source_labels: Sequence[str],
    positions_mm: np.ndarray,
    offset_mm: float,
) -> dict[str, Any]:
    """Write one tiny interpolation triangle per exact bipolar midpoint."""
    from simnibs.mesh_tools import mesh_io

    path = Path(path)
    labels = list(source_labels)
    positions = np.asarray(positions_mm, dtype=float)
    if positions.shape != (len(labels), 3) or len(labels) < 1:
        raise ValueError("Source names and positions are incompatible")
    if len(labels) != len(set(labels)):
        raise ValueError("Source labels must be unique")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for index, point in enumerate(positions):
        triangle, face = source_triangle_vertices(point, offset_mm)
        vertices.append(triangle)
        # SimNIBS/Gmsh element indices are one-based.
        faces.append(face + 3 * index + 1)
    mesh = mesh_io.Msh(
        nodes=np.vstack(vertices), elements={"triangles": np.vstack(faces)}
    )
    mesh.elm.tag1 = np.arange(1, len(labels) + 1, dtype=np.int32)
    mesh.elm.tag2 = mesh.elm.tag1.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_io.write_msh(mesh, str(path))
    return {
        "source_count": len(labels),
        "node_count": int(mesh.nodes.nr),
        "triangle_count": int(mesh.elm.nr),
        "first_vertex_indices_zero_based": (3 * np.arange(len(labels))).tolist(),
    }


def prepare_subject_forward_inputs(
    dataset_root: str | Path,
    subject: str,
    working_dir: str | Path,
    source_triangle_offset_mm: float,
) -> dict[str, Any]:
    """Create ordered scalp/source inputs without reading any signal samples."""
    geometry = _subject_geometry_headers(Path(dataset_root), subject)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    eeg_names = list(geometry["eeg_positions_mm"])
    eeg_positions = np.vstack([geometry["eeg_positions_mm"][name] for name in eeg_names])
    # The solver omits one reference solve and reconstructs it before applying
    # an average reference. Choosing the first released coordinate is fixed and
    # has no effect on the average-referenced forward operator.
    reference_name = eeg_names[0]
    cap_path = write_simnibs_eeg_cap(
        working_dir / f"{subject}_native_t1_eeg.csv",
        eeg_names,
        eeg_positions,
        reference_name,
    )

    source_names: list[str] = []
    source_positions_m: list[np.ndarray] = []
    source_orientations: list[np.ndarray] = []
    orientation_methods: list[str] = []
    for label, position in geometry["ieeg_positions_m"].items():
        try:
            orientation, method = infer_bipolar_orientation(
                label, geometry["ieeg_positions_m"]
            )
        except ValueError:
            continue
        source_names.append(label)
        source_positions_m.append(position)
        source_orientations.append(orientation)
        orientation_methods.append(method)
    if not source_names:
        raise ValueError(f"{subject}: no orientable bipolar source positions")
    positions_m = np.vstack(source_positions_m)
    orientations = np.vstack(source_orientations)
    source_mesh_path = working_dir / f"{subject}_native_t1_bipolar_sources.msh"
    mesh_summary = write_simnibs_source_mesh(
        source_mesh_path,
        source_names,
        positions_m * 1000.0,
        source_triangle_offset_mm,
    )

    manifest = {
        "subject": subject,
        "coordinate_system": "native_T1w_world_mm",
        "eeg_cap_path": str(cap_path),
        "eeg_channel_names": eeg_names,
        "eeg_reference_name": reference_name,
        "eeg_positions_mm": eeg_positions.tolist(),
        "source_mesh_path": str(source_mesh_path),
        "source_labels": source_names,
        "source_positions_mm": (positions_m * 1000.0).tolist(),
        "source_orientations_lower_to_higher_contact": orientations.tolist(),
        "source_orientation_methods": orientation_methods,
        "source_trace_sign_convention": (
            "released bipolar trace a-b multiplies the unit vector from lower "
            "to higher contact number without data-dependent sign selection"
        ),
        **mesh_summary,
    }
    manifest_path = working_dir / f"{subject}_forward_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def reconstruct_average_referenced_eeg_forward(
    reciprocal_leadfield: np.ndarray,
    electrode_names: Sequence[str],
    reference_name: str,
) -> np.ndarray:
    """Apply reciprocity, restore the reference, and average-reference sensors."""
    reciprocal = np.asarray(reciprocal_leadfield, dtype=float)
    names = list(electrode_names)
    if reciprocal.ndim != 3 or reciprocal.shape[0] != len(names) - 1:
        raise ValueError("Reciprocal lead field must omit exactly one reference row")
    if reference_name not in names:
        raise ValueError("Reference electrode is absent from electrode_names")
    reference_index = names.index(reference_name)
    # SimNIBS documents the negative sign when converting TES reciprocity to EEG.
    eeg = -np.insert(
        reciprocal,
        reference_index,
        np.zeros((1, reciprocal.shape[1], reciprocal.shape[2])),
        axis=0,
    )
    eeg -= eeg.mean(axis=0, keepdims=True)
    return eeg


def oriented_source_leadfield(
    eeg_forward_xyz: np.ndarray,
    first_vertex_indices: Sequence[int],
    source_orientations: np.ndarray,
) -> np.ndarray:
    """Select exact midpoint vertices and contract XYZ with fixed orientations."""
    forward = np.asarray(eeg_forward_xyz, dtype=float)
    indices = np.asarray(first_vertex_indices, dtype=int)
    orientations = np.asarray(source_orientations, dtype=float)
    if forward.ndim != 3 or forward.shape[2] != 3:
        raise ValueError("EEG forward must have shape channels x nodes x 3")
    if orientations.shape != (len(indices), 3):
        raise ValueError("Source orientations and indices are incompatible")
    if np.any(indices < 0) or np.any(indices >= forward.shape[1]):
        raise ValueError("A source vertex index is outside the forward solution")
    norms = np.linalg.norm(orientations, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-8):
        raise ValueError("Source orientations must be unit vectors")
    selected = forward[:, indices, :]
    return np.einsum("cso,so->cs", selected, orientations)


def source_depths_to_scalp(
    mesh_path: str | Path, source_positions_mm: np.ndarray
) -> np.ndarray:
    """Return exact Euclidean distances to the CHARM scalp surface."""
    from simnibs.mesh_tools import mesh_io
    from simnibs.utils.mesh_element_properties import ElementTags

    positions = np.asarray(source_positions_mm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("source_positions_mm must be a finite N x 3 matrix")
    mesh = mesh_io.read_msh(str(mesh_path))
    surface = _tagged_surface(mesh, ElementTags.SCALP_TH_SURFACE)
    _, _, projections, _ = surface.project_points(positions)
    depths = np.linalg.norm(positions - projections, axis=1)
    if not np.all(np.isfinite(depths)) or np.any(depths <= 0):
        raise RuntimeError("Invalid source-to-scalp depth calculation")
    return depths


def _tagged_surface(mesh, surface_tag: int):
    """Build a cortech surface with only its own reindexed surface vertices."""
    import cortech

    cropped = mesh.crop_mesh(tags=[surface_tag])
    if cropped.elm.nr < 1 or cropped.nodes.nr < 3:
        raise RuntimeError(f"Mesh surface tag {surface_tag} is empty")
    faces = cropped.elm.node_number_list[:, :3] - 1
    if np.any(faces < 0) or np.any(faces >= cropped.nodes.nr):
        raise RuntimeError(f"Mesh surface tag {surface_tag} has invalid node indices")
    if np.unique(faces).size != cropped.nodes.nr:
        raise RuntimeError(f"Mesh surface tag {surface_tag} retains unused nodes")
    return cortech.Surface(cropped.nodes.node_coord, faces)


def _standard_conductivities(config: dict[str, Any], skull_conductivity: float):
    from simnibs.utils import cond_utils

    conductivities = cond_utils.standard_cond()
    # Python index is tissue tag minus one. CHARM 4.6 represents skull using
    # compact/spongy tags 7/8; tag 4 is retained for compatibility with meshes
    # that use a single bone compartment. All three are treated as skull.
    conductivities[0].value = float(config["head_model"]["brain_conductivity_s_per_m"])
    conductivities[1].value = float(config["head_model"]["brain_conductivity_s_per_m"])
    conductivities[2].value = float(config["head_model"]["csf_conductivity_s_per_m"])
    conductivities[3].value = float(skull_conductivity)
    conductivities[4].value = float(config["head_model"]["scalp_conductivity_s_per_m"])
    conductivities[6].value = float(skull_conductivity)
    conductivities[7].value = float(skull_conductivity)
    return conductivities


def compute_subject_fem_leadfield(
    config: dict[str, Any], subject: str, skull_conductivity: float, cpus: int = 1
) -> tuple[dict[str, Any], Path]:
    """Run a reciprocal FEM solve with residual checks and save the oriented lead field."""
    import h5py
    from simnibs.eeg.forward import compute_tdcs_leadfield

    if cpus < 1:
        raise ValueError("cpus must be positive")
    allowed = [float(value) for value in config["head_model"]["skull_conductivities_s_per_m"]]
    if float(skull_conductivity) not in allowed:
        raise ValueError("Skull conductivity is not one of the configured values")
    cache = Path(config["cache_root"])
    subject_dir = cache / "segmentations" / f"m2m_{subject}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Missing CHARM directory: {subject_dir}")
    validation = config["fem_validation"]
    conductivity_label = f"{float(skull_conductivity):.4f}"
    task_dir = cache / "leadfields" / subject / f"skull_{conductivity_label}"
    task_dir.mkdir(parents=True, exist_ok=True)
    completed_attempts = [
        attempt
        for attempt in sorted(task_dir.glob("attempt_*"))
        if (attempt / f"{subject}_oriented_leadfield.npz").is_file()
        and (attempt / f"{subject}_leadfield_record.json").is_file()
        and (attempt / f"{subject}_solver_log.npz").is_file()
    ]
    if completed_attempts:
        raise FileExistsError(
            "A completed lead field already exists: "
            + ", ".join(str(path) for path in completed_attempts)
        )
    attempt_id = (
        datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S%fZ")
        + f"_pid{os.getpid()}"
    )
    fem_dir = task_dir / attempt_id
    fem_dir.mkdir(parents=False, exist_ok=False)
    working = fem_dir / "forward_inputs"
    manifest = prepare_subject_forward_inputs(
        config["dataset"]["root"],
        subject,
        working,
        float(config["head_model"]["source_triangle_offset_mm"]),
    )
    compact_path = fem_dir / f"{subject}_oriented_leadfield.npz"
    record_path = fem_dir / f"{subject}_leadfield_record.json"
    log_jsonl = fem_dir / f"{subject}_solver_log.jsonl"
    log_npz = fem_dir / f"{subject}_solver_log.npz"
    with logged_simnibs_solver(log_jsonl, validation, cpus):
        hdf5_path = compute_tdcs_leadfield(
            subject_dir,
            fem_dir,
            manifest["eeg_cap_path"],
            point_electrodes=True,
            init_kwargs={
                "interpolation": [manifest["source_mesh_path"]],
                "interpolation_tissue": list(
                    config["head_model"]["source_interpolation_tissues"]
                ),
                "tissues": [],
                "cond": _standard_conductivities(config, skull_conductivity),
                "solver_options": "hypre",
            },
            run_kwargs={"save_mat": False, "cpus": int(cpus)},
        )
    with h5py.File(hdf5_path, "r") as stream:
        dataset = stream["mesh_leadfield"]["leadfields"]["tdcs_leadfield"]
        reciprocal = dataset[:]
        electrode_names = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in dataset.attrs["electrode_names"].tolist()
        ]
        reference = dataset.attrs["reference_electrode"]
        if isinstance(reference, bytes):
            reference = reference.decode()
        reference = str(reference)
        units = dataset.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode()
        field = dataset.attrs["field"]
        if isinstance(field, bytes):
            field = field.decode()
        data_type = dataset.attrs["d_type"]
        if isinstance(data_type, bytes):
            data_type = data_type.decode()

    if str(units) != "V/m" or str(field) != "E" or str(data_type) != "node_data":
        raise RuntimeError("Unexpected SimNIBS lead-field units, field, or data type")
    if reference != manifest["eeg_reference_name"] or electrode_names[0] != reference:
        raise RuntimeError("SimNIBS reference electrode differs from the input manifest")
    solver_validation = finalize_solver_log(
        log_jsonl, log_npz, electrode_names[1:], validation
    )

    full_forward = reconstruct_average_referenced_eeg_forward(
        reciprocal, electrode_names, reference
    )
    first_vertices = np.asarray(manifest["first_vertex_indices_zero_based"], dtype=int)
    orientations = np.asarray(
        manifest["source_orientations_lower_to_higher_contact"], dtype=float
    )
    oriented = oriented_source_leadfield(full_forward, first_vertices, orientations)
    source_depths_mm = source_depths_to_scalp(
        subject_dir / f"{subject}.msh",
        np.asarray(manifest["source_positions_mm"], dtype=float),
    )
    if electrode_names != manifest["eeg_channel_names"]:
        raise RuntimeError("SimNIBS electrode ordering differs from the input manifest")

    np.savez_compressed(
        compact_path,
        oriented_lead_field=oriented,
        eeg_channel_names=np.asarray(electrode_names, dtype=str),
        source_labels=np.asarray(manifest["source_labels"], dtype=str),
        source_positions_mm=np.asarray(manifest["source_positions_mm"], dtype=float),
        source_orientations=np.asarray(
            manifest["source_orientations_lower_to_higher_contact"], dtype=float
        ),
        source_depth_mm=source_depths_mm,
    )
    compact_validation = validate_compact_fem_artifact(compact_path, validation)
    manifest_path = Path(working) / f"{subject}_forward_input_manifest.json"
    record = {
        "subject": subject,
        "fem_attempt_id": attempt_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simnibs_version": config["software"]["head_model_version"],
        "head_model": config["head_model"]["kind"],
        "skull_conductivity_s_per_m": float(skull_conductivity),
        "conductivities_s_per_m": {
            "wm": float(config["head_model"]["brain_conductivity_s_per_m"]),
            "gm": float(config["head_model"]["brain_conductivity_s_per_m"]),
            "csf": float(config["head_model"]["csf_conductivity_s_per_m"]),
            "bone_tag_4": float(skull_conductivity),
            "compact_bone_tag_7": float(skull_conductivity),
            "spongy_bone_tag_8": float(skull_conductivity),
            "scalp": float(config["head_model"]["scalp_conductivity_s_per_m"]),
        },
        "point_electrodes": True,
        "reciprocity_sign_applied": True,
        "average_reference_applied": True,
        "interpolation_tissues": config["head_model"]["source_interpolation_tissues"],
        "source_count": len(manifest["source_labels"]),
        "source_depth_mm": {
            "minimum": float(np.min(source_depths_mm)),
            "median": float(np.median(source_depths_mm)),
            "maximum": float(np.max(source_depths_mm)),
        },
        "eeg_channel_count": len(electrode_names),
        "hdf5_path": str(hdf5_path),
        "forward_input_manifest_path": str(manifest_path),
        "compact_path": str(compact_path),
        "hdf5_units": str(units),
        "compact_validation": compact_validation,
        "solver_validation": solver_validation,
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record, compact_path


def tetrahedral_numerical_quality(
    node_coordinates_mm: np.ndarray,
    tetrahedron_node_indices: np.ndarray,
    chunk_size: int = 200_000,
) -> dict[str, Any]:
    """Summarize shape and reference-element conditioning of tetrahedra.

    The mean-ratio and dihedral metrics are invariant to scale.  The reported
    Jacobian condition number is the 2-norm condition number of the affine map
    from a regular reference tetrahedron to each physical element.  It is a
    local FEM conditioning diagnostic, not an estimate of the full assembled
    system's condition number.
    """
    nodes = np.asarray(node_coordinates_mm, dtype=float)
    tetrahedra = np.asarray(tetrahedron_node_indices, dtype=int)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.all(np.isfinite(nodes)):
        raise ValueError("node_coordinates_mm must be a finite N x 3 matrix")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) < 1:
        raise ValueError("tetrahedron_node_indices must be a nonempty M x 4 matrix")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= len(nodes)):
        raise ValueError("A tetrahedron node index is outside node_coordinates_mm")

    reference_edges = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 6.0],
            [0.0, 0.0, np.sqrt(2.0 / 3.0)],
        ]
    )
    inverse_reference_edges = np.linalg.inv(reference_edges)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    # Each tuple gives a face and the vertex opposite it. The normal is flipped
    # away from that vertex, making pi-angle(normal_i, normal_j) the internal
    # dihedral angle for every pair of faces.
    faces = ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3))
    face_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    mean_ratios: list[np.ndarray] = []
    edge_aspects: list[np.ndarray] = []
    minimum_dihedrals: list[np.ndarray] = []
    maximum_dihedrals: list[np.ndarray] = []
    jacobian_conditions: list[np.ndarray] = []

    for start in range(0, len(tetrahedra), chunk_size):
        points = nodes[tetrahedra[start : start + chunk_size]]
        edge_lengths = np.stack(
            [np.linalg.norm(points[:, j] - points[:, i], axis=1) for i, j in edge_pairs],
            axis=1,
        )
        edge_squares_sum = np.sum(edge_lengths**2, axis=1)
        physical_edges = np.stack(
            [
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ],
            axis=2,
        )
        determinants = np.abs(np.linalg.det(physical_edges))
        volumes = determinants / 6.0
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_ratio = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / edge_squares_sum
            edge_aspect = np.max(edge_lengths, axis=1) / np.min(
                edge_lengths, axis=1
            )
            affine_jacobian = physical_edges @ inverse_reference_edges
            singular_values = np.linalg.svd(affine_jacobian, compute_uv=False)
            jacobian_condition = singular_values[:, 0] / singular_values[:, -1]

        outward_normals = []
        for a, b, c, opposite in faces:
            normal = np.cross(points[:, b] - points[:, a], points[:, c] - points[:, a])
            points_inward = np.einsum(
                "ij,ij->i", normal, points[:, opposite] - points[:, a]
            ) > 0
            normal[points_inward] *= -1.0
            norm = np.linalg.norm(normal, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                normal = normal / norm[:, None]
            outward_normals.append(normal)
        dihedrals = np.stack(
            [
                np.pi
                - np.arccos(
                    np.clip(
                        np.einsum(
                            "ij,ij->i", outward_normals[first], outward_normals[second]
                        ),
                        -1.0,
                        1.0,
                    )
                )
                for first, second in face_pairs
            ],
            axis=1,
        )
        mean_ratios.append(mean_ratio)
        edge_aspects.append(edge_aspect)
        minimum_dihedrals.append(np.degrees(np.min(dihedrals, axis=1)))
        maximum_dihedrals.append(np.degrees(np.max(dihedrals, axis=1)))
        jacobian_conditions.append(jacobian_condition)

    mean_ratio = np.concatenate(mean_ratios)
    edge_aspect = np.concatenate(edge_aspects)
    minimum_dihedral = np.concatenate(minimum_dihedrals)
    maximum_dihedral = np.concatenate(maximum_dihedrals)
    jacobian_condition = np.concatenate(jacobian_conditions)
    finite = (
        np.isfinite(mean_ratio)
        & np.isfinite(edge_aspect)
        & np.isfinite(minimum_dihedral)
        & np.isfinite(maximum_dihedral)
        & np.isfinite(jacobian_condition)
    )
    finite_count = int(np.sum(finite))
    if finite_count < 1:
        raise RuntimeError("No tetrahedron has finite numerical-quality metrics")

    def summary(values: np.ndarray) -> dict[str, float]:
        values = values[finite]
        return {
            "minimum": float(np.min(values)),
            "p001": float(np.quantile(values, 0.001)),
            "p01": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "maximum": float(np.max(values)),
        }

    return {
        "tetrahedron_count": int(len(tetrahedra)),
        "finite_metric_count": finite_count,
        "nonfinite_metric_count": int(len(tetrahedra) - finite_count),
        "mean_ratio": summary(mean_ratio),
        "fraction_mean_ratio_below_0_01": float(np.mean(mean_ratio < 0.01)),
        "minimum_dihedral_angle_degrees": summary(minimum_dihedral),
        "maximum_dihedral_angle_degrees": summary(maximum_dihedral),
        "edge_aspect_ratio": summary(edge_aspect),
        "regular_reference_jacobian_condition_number": summary(jacobian_condition),
        "conditioning_interpretation": (
            "local 2-norm condition of the regular-reference affine element map; "
            "not the assembled FEM matrix condition number"
        ),
    }


def electrode_projection_quality(
    electrode_names: Sequence[str],
    raw_positions_mm: np.ndarray,
    projected_positions_mm: np.ndarray,
    scalp_vertices_mm: np.ndarray,
    landmark_positions_mm: dict[str, np.ndarray],
    projected_landmarks_mm: dict[str, np.ndarray],
    nearest_neighbour_count: int,
) -> dict[str, Any]:
    """Measure radial, angular, topological, and fiducial cap alignment."""
    from scipy.spatial import cKDTree

    names = list(electrode_names)
    raw = np.asarray(raw_positions_mm, dtype=float)
    projected = np.asarray(projected_positions_mm, dtype=float)
    scalp_vertices = np.asarray(scalp_vertices_mm, dtype=float)
    if raw.shape != (len(names), 3) or projected.shape != raw.shape:
        raise ValueError("Electrode names and raw/projected coordinates are incompatible")
    if scalp_vertices.ndim != 2 or scalp_vertices.shape[1] != 3:
        raise ValueError("scalp_vertices_mm must be an N x 3 matrix")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(projected)):
        raise ValueError("Electrode coordinates must be finite")
    k = int(nearest_neighbour_count)
    if k < 1 or len(names) <= k:
        raise ValueError("nearest_neighbour_count must be smaller than electrode count")
    if set(landmark_positions_mm) != set(projected_landmarks_mm):
        raise ValueError("Raw and projected landmark names differ")

    center = np.median(scalp_vertices, axis=0)
    displacement = projected - raw
    projection_distances = np.linalg.norm(displacement, axis=1)
    projected_radial = projected - center
    raw_radial = raw - center
    projected_radius = np.linalg.norm(projected_radial, axis=1)
    raw_radius = np.linalg.norm(raw_radial, axis=1)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    denominator = displacement_norm * projected_radius
    radial_alignment = np.ones(len(raw), dtype=float)
    nonzero = denominator > np.finfo(float).eps
    radial_alignment[nonzero] = np.einsum(
        "ij,ij->i", displacement[nonzero], projected_radial[nonzero]
    ) / denominator[nonzero]
    radial_alignment = np.clip(radial_alignment, -1.0, 1.0)
    outward = (radial_alignment >= 0.0) | (~nonzero)
    angular_denominator = raw_radius * projected_radius
    if np.any(angular_denominator <= np.finfo(float).eps):
        raise RuntimeError("An electrode lies at the estimated scalp centre")
    angular = np.degrees(
        np.arccos(
            np.clip(
                np.einsum("ij,ij->i", raw_radial, projected_radial)
                / angular_denominator,
                -1.0,
                1.0,
            )
        )
    )
    radial_scale_ratio = raw_radius / projected_radius

    raw_neighbours = cKDTree(raw).query(raw, k=k + 1)[1][:, 1:]
    projected_query = cKDTree(projected).query(projected, k=k + 1)
    projected_neighbour_distances = projected_query[0][:, 1:]
    projected_neighbours = projected_query[1][:, 1:]
    overlap = np.asarray(
        [
            len(set(raw_row.tolist()) & set(projected_row.tolist())) / k
            for raw_row, projected_row in zip(raw_neighbours, projected_neighbours)
        ],
        dtype=float,
    )
    raw_edges = np.linalg.norm(raw[:, None, :] - raw[raw_neighbours], axis=2)
    projected_edges = np.linalg.norm(
        projected[:, None, :] - projected[raw_neighbours], axis=2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        neighbour_edge_ratio = projected_edges / raw_edges

    landmark_distance_by_name = {
        name: float(
            np.linalg.norm(
                np.asarray(landmark_positions_mm[name], dtype=float)
                - np.asarray(projected_landmarks_mm[name], dtype=float)
            )
        )
        for name in sorted(landmark_positions_mm)
    }

    return {
        "estimated_scalp_center_mm": center.tolist(),
        "projection_distance_mm": {
            "median": float(np.median(projection_distances)),
            "p95": float(np.quantile(projection_distances, 0.95)),
            "maximum": float(np.max(projection_distances)),
        },
        "outward_projection_fraction": float(np.mean(outward)),
        "radial_alignment_cosine": {
            "median": float(np.median(radial_alignment)),
            "p05": float(np.quantile(radial_alignment, 0.05)),
            "minimum": float(np.min(radial_alignment)),
        },
        "radial_alignment_absolute_cosine": {
            "median": float(np.median(np.abs(radial_alignment))),
            "p05": float(np.quantile(np.abs(radial_alignment), 0.05)),
            "minimum": float(np.min(np.abs(radial_alignment))),
        },
        "angular_displacement_degrees": {
            "median": float(np.median(angular)),
            "p95": float(np.quantile(angular, 0.95)),
            "maximum": float(np.max(angular)),
        },
        "nearest_neighbour_count": k,
        "neighbour_overlap_fraction": {
            "mean": float(np.mean(overlap)),
            "minimum": float(np.min(overlap)),
        },
        "neighbour_edge_length_ratio_projected_over_raw": {
            "median": float(np.nanmedian(neighbour_edge_ratio)),
            "p05": float(np.nanquantile(neighbour_edge_ratio, 0.05)),
            "p95": float(np.nanquantile(neighbour_edge_ratio, 0.95)),
        },
        "minimum_projected_interelectrode_distance_mm": float(
            np.min(projected_neighbour_distances)
        ),
        "centroid_projection_displacement_mm": float(
            np.linalg.norm(np.mean(projected, axis=0) - np.mean(raw, axis=0))
        ),
        "radial_scale_ratio_raw_over_projected": {
            "median": float(np.median(radial_scale_ratio)),
            "p05": float(np.quantile(radial_scale_ratio, 0.05)),
            "p95": float(np.quantile(radial_scale_ratio, 0.95)),
        },
        "fiducial_to_scalp_distance_mm_by_name": landmark_distance_by_name,
        "maximum_fiducial_to_scalp_distance_mm": float(
            max(landmark_distance_by_name.values())
        ),
        "distance_mm_by_electrode": {
            name: float(value) for name, value in zip(names, projection_distances)
        },
        "angular_displacement_degrees_by_electrode": {
            name: float(value) for name, value in zip(names, angular)
        },
        "radial_alignment_cosine_by_electrode": {
            name: float(value) for name, value in zip(names, radial_alignment)
        },
    }


def automated_head_model_qc(
    config: dict[str, Any], subject: str
) -> dict[str, Any]:
    """Check CHARM compartments, tetrahedra, electrode fit, and source proximity."""
    from scipy.spatial import cKDTree
    from simnibs.mesh_tools import mesh_io
    from simnibs.utils.mesh_element_properties import ElementTags

    cache = Path(config["cache_root"])
    subject_dir = cache / "segmentations" / f"m2m_{subject}"
    mesh_path = subject_dir / f"{subject}.msh"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing CHARM mesh: {mesh_path}")
    mesh = mesh_io.read_msh(str(mesh_path))
    geometry = _subject_geometry_headers(Path(config["dataset"]["root"]), subject)

    tetra_numbers = mesh.elm.tetrahedra
    tetra_indices = tetra_numbers - 1
    tetra_tags = mesh.elm.tag1[tetra_indices]
    tissue_tag_groups = {
        "wm": [ElementTags.WM],
        "gm": [ElementTags.GM],
        "csf": [ElementTags.CSF],
        "skull": [
            ElementTags.BONE,
            ElementTags.COMPACT_BONE,
            ElementTags.SPONGY_BONE,
        ],
        "scalp": [ElementTags.SCALP],
    }
    tissue_counts = {
        name: int(np.sum(np.isin(tetra_tags, tags)))
        for name, tags in tissue_tag_groups.items()
    }
    tetrahedron_counts_by_tag = {
        str(int(tag)): int(count)
        for tag, count in zip(*np.unique(tetra_tags, return_counts=True))
    }
    volumes = np.asarray(mesh.elements_volumes_and_areas()[tetra_numbers], dtype=float)
    tetra_nodes = mesh.elm.node_number_list[tetra_indices, :4]
    distinct_nodes = np.array([len(set(row)) == 4 for row in tetra_nodes])
    invalid = (~np.isfinite(volumes)) | (volumes <= np.finfo(float).tiny) | (~distinct_nodes)

    surface = _tagged_surface(mesh, ElementTags.SCALP_TH_SURFACE)
    eeg_names = list(geometry["eeg_positions_mm"])
    eeg_positions = np.vstack([geometry["eeg_positions_mm"][name] for name in eeg_names])
    _, _, projections, _ = surface.project_points(eeg_positions)
    eeg_distances = np.linalg.norm(eeg_positions - projections, axis=1)

    source_names = list(geometry["ieeg_positions_m"])
    source_positions = np.vstack(
        [geometry["ieeg_positions_m"][name] for name in source_names]
    ) * 1000.0
    containing = mesh.find_tetrahedron_with_points(
        source_positions, compute_baricentric=False
    )
    containing_tags = np.full(len(source_names), -1, dtype=int)
    inside = containing > 0
    containing_tags[inside] = mesh.elm.tag1[containing[inside] - 1]
    brain_mesh = mesh.crop_mesh(tags=[1, 2])
    tree = cKDTree(brain_mesh.nodes.node_coord)
    conservative_node_distances = tree.query(source_positions, k=1)[0]
    source_distances = np.where(
        np.isin(containing_tags, [1, 2]), 0.0, conservative_node_distances
    )

    mesh_quality = tetrahedral_numerical_quality(
        mesh.nodes.node_coord, tetra_nodes - 1
    )
    landmark_names = list(geometry["anatomical_landmarks_mm"])
    landmark_positions = np.vstack(
        [geometry["anatomical_landmarks_mm"][name] for name in landmark_names]
    )
    _, _, landmark_projections, _ = surface.project_points(landmark_positions)
    projected_landmarks = {
        name: position for name, position in zip(landmark_names, landmark_projections)
    }
    qc_settings = config["head_model_qc"]
    electrode_thresholds = qc_settings["electrode_projection"]
    mesh_thresholds = qc_settings["mesh_numerical"]
    electrode_quality = electrode_projection_quality(
        eeg_names,
        eeg_positions,
        projections,
        surface.vertices,
        geometry["anatomical_landmarks_mm"],
        projected_landmarks,
        int(electrode_thresholds["nearest_neighbour_count"]),
    )
    radial_cutoff = float(
        electrode_thresholds["minimum_individual_radial_alignment_absolute_cosine"]
    )
    radial_values = np.asarray(
        list(electrode_quality["radial_alignment_cosine_by_electrode"].values())
    )
    electrode_quality["radially_aligned_projection_fraction"] = float(
        np.mean(np.abs(radial_values) >= radial_cutoff)
    )
    electrode_quality["radially_aligned_projection_absolute_cosine_cutoff"] = (
        radial_cutoff
    )
    display_indices = np.linspace(
        0, len(surface.vertices) - 1, min(10_000, len(surface.vertices)), dtype=int
    )

    median_eeg = float(np.median(eeg_distances))
    p95_eeg = float(np.quantile(eeg_distances, 0.95))
    max_source = float(np.max(source_distances))
    gates = config["eligibility"]
    checks = {
        "all_five_tissue_compartments_present": all(
            tissue_counts[name] > 0 for name in tissue_counts
        ),
        "no_invalid_tetrahedra": int(np.sum(invalid)) == 0,
        "mesh_quality_metrics_complete": mesh_quality["nonfinite_metric_count"] == 0,
        "mesh_mean_ratio_tail_pass": mesh_quality["mean_ratio"]["p001"]
        >= float(mesh_thresholds["minimum_mean_ratio_p001"]),
        "mesh_low_mean_ratio_fraction_pass": mesh_quality[
            "fraction_mean_ratio_below_0_01"
        ]
        <= float(mesh_thresholds["maximum_fraction_mean_ratio_below_0_01"]),
        "mesh_minimum_dihedral_tail_pass": mesh_quality[
            "minimum_dihedral_angle_degrees"
        ]["p001"]
        >= float(mesh_thresholds["minimum_dihedral_angle_p001_degrees"]),
        "mesh_maximum_dihedral_tail_pass": mesh_quality[
            "maximum_dihedral_angle_degrees"
        ]["p999"]
        <= float(mesh_thresholds["maximum_dihedral_angle_p999_degrees"]),
        "mesh_edge_aspect_tail_pass": mesh_quality["edge_aspect_ratio"]["p999"]
        <= float(mesh_thresholds["maximum_edge_aspect_ratio_p999"]),
        "fem_element_jacobian_conditioning_pass": mesh_quality[
            "regular_reference_jacobian_condition_number"
        ]["p999"]
        <= float(mesh_thresholds["maximum_jacobian_condition_number_p999"]),
        "electrode_projection_median_distance_pass": electrode_quality[
            "projection_distance_mm"
        ]["median"]
        <= float(electrode_thresholds["maximum_median_projection_distance_mm"]),
        "electrode_projection_p95_distance_pass": electrode_quality[
            "projection_distance_mm"
        ]["p95"]
        <= float(electrode_thresholds["maximum_p95_projection_distance_mm"]),
        "electrode_projection_maximum_distance_pass": electrode_quality[
            "projection_distance_mm"
        ]["maximum"]
        <= float(electrode_thresholds["maximum_projection_distance_mm"]),
        "electrode_projection_radially_aligned_fraction_pass": electrode_quality[
            "radially_aligned_projection_fraction"
        ]
        >= float(
            electrode_thresholds["minimum_radially_aligned_projection_fraction"]
        ),
        "electrode_projection_radial_alignment_pass": electrode_quality[
            "radial_alignment_absolute_cosine"
        ]["median"]
        >= float(
            electrode_thresholds["minimum_median_radial_alignment_absolute_cosine"]
        ),
        "electrode_projection_angular_displacement_pass": electrode_quality[
            "angular_displacement_degrees"
        ]["p95"]
        <= float(electrode_thresholds["maximum_p95_angular_displacement_degrees"]),
        "electrode_neighbour_topology_pass": electrode_quality[
            "neighbour_overlap_fraction"
        ]["mean"]
        >= float(electrode_thresholds["minimum_mean_neighbour_overlap_fraction"]),
        "electrode_projected_separation_pass": electrode_quality[
            "minimum_projected_interelectrode_distance_mm"
        ]
        >= float(electrode_thresholds["minimum_projected_interelectrode_distance_mm"]),
        "fiducial_to_scalp_distance_pass": electrode_quality[
            "maximum_fiducial_to_scalp_distance_mm"
        ]
        <= float(electrode_thresholds["maximum_fiducial_to_scalp_distance_mm"]),
        "electrode_centroid_displacement_pass": electrode_quality[
            "centroid_projection_displacement_mm"
        ]
        <= float(electrode_thresholds["maximum_centroid_projection_displacement_mm"]),
        "electrode_radial_scale_pass": electrode_quality[
            "radial_scale_ratio_raw_over_projected"
        ]["median"]
        <= float(electrode_thresholds["maximum_median_radial_scale_ratio"]),
        "source_distance_map_complete": bool(np.all(np.isfinite(source_distances))),
        "charm_report_html_present": (subject_dir / "charm_report.html").is_file(),
    }
    return {
        "subject": subject,
        "mesh_path": str(mesh_path),
        "mesh_node_count": int(mesh.nodes.nr),
        "tetrahedron_count": int(len(tetra_numbers)),
        "tissue_tetrahedron_counts": tissue_counts,
        "tetrahedron_counts_by_tag": tetrahedron_counts_by_tag,
        "invalid_tetrahedron_count": int(np.sum(invalid)),
        "minimum_tetrahedron_volume_mm3": float(np.min(volumes)),
        "tetrahedral_numerical_quality": mesh_quality,
        "head_model_qc_thresholds": qc_settings,
        "eeg_electrode_count": len(eeg_names),
        "eeg_to_scalp_distance_mm": {
            "median": median_eeg,
            "p95": p95_eeg,
            "maximum": float(np.max(eeg_distances)),
        },
        "electrode_projection_qc": electrode_quality,
        "electrode_projection_geometry": {
            "electrode_names": eeg_names,
            "raw_positions_mm": eeg_positions.tolist(),
            "projected_positions_mm": projections.tolist(),
            "landmark_names": landmark_names,
            "raw_landmark_positions_mm": landmark_positions.tolist(),
            "projected_landmark_positions_mm": landmark_projections.tolist(),
            "scalp_display_vertices_mm": surface.vertices[display_indices].tolist(),
        },
        "ieeg_source_count": len(source_names),
        "source_to_brain_conservative_node_distance_mm": {
            "median": float(np.median(source_distances)),
            "p95": float(np.quantile(source_distances, 0.95)),
            "maximum": max_source,
        },
        "source_distance_mm_by_label": {
            name: float(distance)
            for name, distance in zip(source_names, source_distances)
        },
        "source_within_brain_distance_count": int(
            np.sum(
                source_distances
                <= float(gates["maximum_source_to_brain_distance_mm"])
            )
        ),
        "checks": checks,
        "automated_qc_pass": all(checks.values()),
    }
