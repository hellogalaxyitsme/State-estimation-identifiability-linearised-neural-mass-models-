"""EEG forward-model construction and independent average-reference coordinates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-neural-observability")

import mne
import numpy as np

from neural_observability.head_models.referencing import average_reference_basis


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


CHANNELS_19 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


@dataclass
class ForwardData:
    """A parcel/source lead field in independent referenced coordinates."""

    lead_field: np.ndarray
    positions_m: np.ndarray
    source_names: list[str]
    metadata: dict[str, Any]


def montage_spec(label: str) -> tuple[mne.channels.DigMontage, list[str]]:
    """Return the requested standard montage and ordered channel list."""
    if label == "19":
        return mne.channels.make_standard_montage("standard_1020"), CHANNELS_19
    if label == "32":
        montage = mne.channels.make_standard_montage("biosemi32")
        return montage, list(montage.ch_names)
    if label == "64":
        montage = mne.channels.make_standard_montage("biosemi64")
        return montage, list(montage.ch_names)
    if label == "128":
        montage = mne.channels.make_standard_montage("biosemi128")
        return montage, list(montage.ch_names)
    if label == "256":
        montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
        return montage, [name for name in montage.ch_names if name != "Cz"]
    raise ValueError(f"Unsupported montage label: {label}")


def _cache_path(
    config: dict[str, Any], project_root: Path, montage: str, kind: str
) -> Path:
    cache_dir = resolve_project_path(project_root, config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    if kind == "fsaverage_bem":
        detail = f"{config['spacing']}_{config['parcellation']}"
    else:
        detail = (
            f"{config['sphere_source_spacing_mm']}_"
            f"{config['sphere_max_sources']}"
        )
    return cache_dir / f"forward_{kind}_{detail}_{montage}_mne-{mne.__version__}.npz"


def _write_cache(path: Path, data: ForwardData) -> None:
    np.savez_compressed(
        path,
        lead_field=data.lead_field,
        positions_m=data.positions_m,
        source_names=np.asarray(data.source_names, dtype=str),
        metadata_json=json.dumps(data.metadata, sort_keys=True),
    )


def _read_cache(path: Path) -> ForwardData:
    with np.load(path, allow_pickle=False) as stored:
        return ForwardData(
            lead_field=stored["lead_field"],
            positions_m=stored["positions_m"],
            source_names=stored["source_names"].tolist(),
            metadata=json.loads(str(stored["metadata_json"])),
        )


def _farthest_point_subset(positions: np.ndarray, count: int) -> np.ndarray:
    """Deterministically choose a spatially dispersed subset."""
    if count >= len(positions):
        return np.arange(len(positions))
    selected = [int(np.argmin(positions[:, 0]))]
    minimum_distance_sq = np.sum((positions - positions[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        candidate = int(np.argmax(minimum_distance_sq))
        selected.append(candidate)
        distance_sq = np.sum((positions - positions[candidate]) ** 2, axis=1)
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
    return np.asarray(sorted(selected), dtype=int)


def build_sphere_forward(
    config: dict[str, Any], project_root: Path, montage_label: str
) -> ForwardData:
    """Construct the explicitly surrogate four-shell spherical forward model."""
    cache = _cache_path(config, project_root, montage_label, "sphere")
    if cache.exists():
        return _read_cache(cache)

    mne.set_log_level("WARNING")
    montage, channel_names = montage_spec(montage_label)
    info = mne.create_info(channel_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=0.09,
        relative_radii=(0.90, 0.92, 0.97, 1.0),
        sigmas=(0.33, 1.79, 0.0042, 0.33),
        verbose=False,
    )
    source_space = mne.setup_volume_source_space(
        subject=None,
        pos=float(config["sphere_source_spacing_mm"]),
        sphere=(0.0, 0.0, 0.0, 0.075),
        exclude=10.0,
        verbose=False,
    )
    forward = mne.make_forward_solution(
        info,
        trans=None,
        src=source_space,
        bem=sphere,
        eeg=True,
        meg=False,
        mindist=0.0,
        n_jobs=1,
        verbose=False,
    )
    free_gain = forward["sol"]["data"]
    positions = np.asarray(forward["source_rr"], dtype=float)
    radial = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    lead_field = np.column_stack(
        [free_gain[:, 3 * index : 3 * index + 3] @ radial[index] for index in range(len(positions))]
    )
    selected = _farthest_point_subset(positions, int(config["sphere_max_sources"]))
    positions = positions[selected]
    lead_field = lead_field[:, selected]
    basis = average_reference_basis(len(channel_names))
    lead_field = basis.T @ lead_field
    metadata: dict[str, Any] = {
        "kind": "sphere",
        "evidence_class": "surrogate",
        "montage": montage_label,
        "physical_channel_count": len(channel_names),
        "independent_referenced_channel_count": len(channel_names) - 1,
        "source_count": len(selected),
        "source_orientation": "radial",
        "source_spacing_mm": float(config["sphere_source_spacing_mm"]),
        "reference": "orthonormal zero-sum basis",
        "mne_version": mne.__version__,
    }
    data = ForwardData(
        lead_field=lead_field,
        positions_m=positions,
        source_names=[f"sphere_{index:03d}" for index in range(len(selected))],
        metadata=metadata,
    )
    _write_cache(cache, data)
    return data


def _fsaverage_assets(config: dict[str, Any], project_root: Path) -> tuple[Path, Path, Path]:
    subjects_dir = resolve_project_path(project_root, config["subjects_dir"])
    fs_dir = subjects_dir / "fsaverage"
    trans = fs_dir / "bem" / "fsaverage-trans.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    missing = [path for path in (fs_dir, trans, bem) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "fsaverage assets are missing. Run `python3 scripts/fetch_fsaverage.py` "
            f"before the primary analysis. Missing: {formatted}"
        )
    return subjects_dir, trans, bem


def build_fsaverage_forward(
    config: dict[str, Any], project_root: Path, montage_label: str
) -> ForwardData:
    """Build an fsaverage BEM forward and reduce it to cortical parcels."""
    cache = _cache_path(config, project_root, montage_label, "fsaverage_bem")
    if cache.exists():
        return _read_cache(cache)

    mne.set_log_level("WARNING")
    subjects_dir, trans_path, bem_path = _fsaverage_assets(config, project_root)
    montage, channel_names = montage_spec(montage_label)
    info = mne.create_info(channel_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    source_space = mne.setup_source_space(
        "fsaverage",
        spacing=str(config["spacing"]),
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    forward = mne.make_forward_solution(
        info,
        trans=trans_path,
        src=source_space,
        bem=bem_path,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    fixed = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        copy=True,
        verbose=False,
    )
    gain = np.asarray(fixed["sol"]["data"], dtype=float)
    source_positions = np.asarray(fixed["source_rr"], dtype=float)
    src = fixed["src"]
    left_vertices = np.asarray(src[0]["vertno"], dtype=int)
    right_vertices = np.asarray(src[1]["vertno"], dtype=int)
    vertex_to_column = {
        "lh": {int(vertex): index for index, vertex in enumerate(left_vertices)},
        "rh": {
            int(vertex): len(left_vertices) + index
            for index, vertex in enumerate(right_vertices)
        },
    }
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=str(config["parcellation"]),
        subjects_dir=subjects_dir,
        verbose=False,
    )
    labels = [
        label
        for label in labels
        if not label.name.lower().startswith(("unknown", "corpuscallosum"))
    ]
    parcel_leads: list[np.ndarray] = []
    parcel_positions: list[np.ndarray] = []
    parcel_names: list[str] = []
    parcel_vertex_counts: list[int] = []
    for label in sorted(labels, key=lambda item: item.name):
        mapping = vertex_to_column[label.hemi]
        columns = [mapping[int(vertex)] for vertex in label.vertices if int(vertex) in mapping]
        if not columns:
            continue
        columns_array = np.asarray(columns, dtype=int)
        if config["patch_normalization"] != "l2":
            raise ValueError("Only l2 patch normalization is implemented")
        weights = np.full(len(columns_array), 1.0 / np.sqrt(len(columns_array)))
        parcel_leads.append(gain[:, columns_array] @ weights)
        parcel_positions.append(np.average(source_positions[columns_array], axis=0, weights=weights**2))
        parcel_names.append(label.name)
        parcel_vertex_counts.append(len(columns_array))

    lead_field = np.column_stack(parcel_leads)
    positions = np.vstack(parcel_positions)
    basis = average_reference_basis(len(channel_names))
    lead_field = basis.T @ lead_field
    metadata = {
        "kind": "fsaverage_bem",
        "evidence_class": "anatomical_template",
        "montage": montage_label,
        "physical_channel_count": len(channel_names),
        "independent_referenced_channel_count": len(channel_names) - 1,
        "source_count": len(parcel_names),
        "source_orientation": "cortical-normal fixed orientation",
        "spacing": config["spacing"],
        "parcellation": config["parcellation"],
        "patch_normalization": config["patch_normalization"],
        "parcel_vertex_count_min": min(parcel_vertex_counts),
        "parcel_vertex_count_median": float(np.median(parcel_vertex_counts)),
        "parcel_vertex_count_max": max(parcel_vertex_counts),
        "reference": "orthonormal zero-sum basis",
        "mne_version": mne.__version__,
    }
    data = ForwardData(
        lead_field=lead_field,
        positions_m=positions,
        source_names=parcel_names,
        metadata=metadata,
    )
    _write_cache(cache, data)
    return data


def load_or_build_forward(
    config: dict[str, Any],
    project_root: Path,
    montage_label: str,
    kind_override: str | None = None,
) -> ForwardData:
    """Load or build one configured forward model."""
    kind = kind_override or str(config["kind"])
    if kind == "sphere":
        return build_sphere_forward(config, project_root, montage_label)
    if kind == "fsaverage_bem":
        return build_fsaverage_forward(config, project_root, montage_label)
    raise ValueError(f"Unsupported forward kind: {kind}")
