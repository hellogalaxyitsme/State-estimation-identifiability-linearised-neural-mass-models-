"""Bipolar intracranial sources and template-head forward models for the working-memory cohort."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np



@dataclass(frozen=True)
class BipolarSource:
    """One adjacent, anatomically labelled intracranial source."""

    name: str
    lower_name: str
    upper_name: str
    lower_index: int
    upper_index: int
    midpoint_mni_mm: tuple[float, float, float]
    direction_mni: tuple[float, float, float]


def adjacent_bipolar_sources(
    channel_names: Sequence[str], electrode_rows: Sequence[dict[str, Any]]
) -> list[BipolarSource]:
    """Return deterministic lower-minus-higher adjacent labelled contacts."""
    names = [str(value).strip() for value in channel_names]
    if len(names) != len(set(names)):
        raise ValueError("Intracranial channel names must be unique")
    index = {name: number for number, name in enumerate(names)}
    electrodes = {str(row["name"]).strip(): row for row in electrode_rows}
    sources: list[BipolarSource] = []
    for lower_name in sorted(names):
        match = re.fullmatch(r"(.+?)(\d+)", lower_name)
        if match is None:
            continue
        upper_name = f"{match.group(1)}{int(match.group(2)) + 1}"
        if upper_name not in index or lower_name not in electrodes or upper_name not in electrodes:
            continue
        lower = electrodes[lower_name]
        upper = electrodes[upper_name]
        lower_label = str(lower.get("AnatomicalLocation", "")).strip()
        upper_label = str(upper.get("AnatomicalLocation", "")).strip()
        invalid_labels = {"", "n/a", "na", "no_label_found"}
        if lower_label.lower() in invalid_labels or upper_label.lower() in invalid_labels:
            continue
        lower_position = np.asarray(
            [float(lower[axis]) for axis in ("x", "y", "z")], dtype=float
        )
        upper_position = np.asarray(
            [float(upper[axis]) for axis in ("x", "y", "z")], dtype=float
        )
        difference = upper_position - lower_position
        norm = float(np.linalg.norm(difference))
        if not np.all(np.isfinite(lower_position)) or not np.all(
            np.isfinite(upper_position)
        ):
            raise ValueError("Electrode coordinates must be finite")
        if norm <= np.finfo(float).tiny:
            raise ValueError("Adjacent contacts cannot occupy the same point")
        midpoint = 0.5 * (lower_position + upper_position)
        direction = difference / norm
        sources.append(
            BipolarSource(
                name=f"{lower_name}-{upper_name}",
                lower_name=lower_name,
                upper_name=upper_name,
                lower_index=index[lower_name],
                upper_index=index[upper_name],
                midpoint_mni_mm=tuple(float(value) for value in midpoint),
                direction_mni=tuple(float(value) for value in direction),
            )
        )
    return sources


def bipolarize_trials(
    contact_trials: np.ndarray, sources: Sequence[BipolarSource]
) -> np.ndarray:
    """Convert trial x contact x sample data to lower-minus-higher sources."""
    values = np.asarray(contact_trials, dtype=float)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("contact_trials must be a finite three-dimensional array")
    if not sources:
        raise ValueError("At least one bipolar source is required")
    lower = [source.lower_index for source in sources]
    upper = [source.upper_index for source in sources]
    if max(lower + upper) >= values.shape[1]:
        raise ValueError("A bipolar source index is out of range")
    return values[:, lower, :] - values[:, upper, :]


def transform_mni_sources_to_head(
    sources: Sequence[BipolarSource], subjects_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Transform MNI positions/directions into fsaverage head coordinates."""
    import mne

    directory = Path(subjects_dir)
    mri_to_mni = mne.read_talxfm("fsaverage", subjects_dir=directory, verbose=False)
    mni_to_mri = mne.transforms.invert_transform(mri_to_mni)
    head_to_mri = mne.read_trans(
        directory / "fsaverage" / "bem" / "fsaverage-trans.fif"
    )
    mri_to_head = mne.transforms.invert_transform(head_to_mri)
    positions_mni = np.asarray([source.midpoint_mni_mm for source in sources]) / 1000.0
    directions_mni = np.asarray([source.direction_mni for source in sources])
    positions_mri = mne.transforms.apply_trans(mni_to_mri, positions_mni)
    positions_head = mne.transforms.apply_trans(mri_to_head, positions_mri)
    combined_rotation = mri_to_head["trans"][:3, :3] @ mni_to_mri["trans"][:3, :3]
    directions_head = directions_mni @ combined_rotation.T
    norms = np.linalg.norm(directions_head, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("A transformed source direction is degenerate")
    directions_head /= norms[:, None]
    if not np.all(np.isfinite(positions_head)) or not np.all(
        np.isfinite(directions_head)
    ):
        raise RuntimeError("The MNI-to-head transform produced invalid values")
    return positions_head, directions_head


def standard_1020_info(channel_names: Sequence[str], sampling_frequency: float):
    """Create MNE info with deterministic legacy 10--20 name handling."""
    import mne

    names = [str(value) for value in channel_names]
    standard = mne.channels.make_standard_montage("standard_1020")
    positions = standard.get_positions()
    aliases = {"T1": "T9", "T2": "T10"}
    channel_positions = {}
    for name in names:
        lookup = aliases.get(name, name)
        if lookup not in positions["ch_pos"]:
            raise ValueError(f"No standard-1020 coordinate for {name}")
        channel_positions[name] = positions["ch_pos"][lookup]
    montage = mne.channels.make_dig_montage(
        ch_pos=channel_positions,
        nasion=positions["nasion"],
        lpa=positions["lpa"],
        rpa=positions["rpa"],
        coord_frame="head",
    )
    info = mne.create_info(names, float(sampling_frequency), ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    return info


def build_template_oriented_forward(
    channel_names: Sequence[str],
    sources: Sequence[BipolarSource],
    subjects_dir: str | Path,
    bem_solution_path: str | Path,
    sampling_frequency: float,
) -> tuple[np.ndarray, list[BipolarSource], dict[str, Any]]:
    """Build the fixed-orientation fsaverage BEM operator, retaining valid columns."""
    import mne

    info = standard_1020_info(channel_names, sampling_frequency)
    positions, directions = transform_mni_sources_to_head(sources, subjects_dir)
    bem = mne.read_bem_solution(bem_solution_path, verbose=False)
    columns = []
    retained: list[BipolarSource] = []
    rejected = []
    for source, position, direction in zip(sources, positions, directions):
        dipole = mne.Dipole(
            times=np.array([0.0]),
            pos=position[None, :],
            amplitude=np.array([1.0]),
            ori=direction[None, :],
            gof=np.array([1.0]),
            name=source.name,
        )
        try:
            forward, _ = mne.make_forward_dipole(
                dipole, bem, info, trans=None, n_jobs=1, verbose=False
            )
            column = np.asarray(forward["sol"]["data"], dtype=float).reshape(
                len(channel_names), -1
            )[:, 0]
            referenced = column - column.mean()
            if not np.all(np.isfinite(referenced)) or np.linalg.norm(
                referenced
            ) <= np.finfo(float).tiny:
                raise RuntimeError("Too few finite nonzero forward columns")
        except Exception as error:  # MNE reports outside-inner-skull sources here.
            rejected.append({"source": source.name, "reason": str(error)})
            continue
        columns.append(column)
        retained.append(source)
    if not columns:
        raise RuntimeError("No template forward column passed")
    return np.column_stack(columns), retained, {
        "requested_source_count": len(sources),
        "retained_source_count": len(retained),
        "rejected_sources": rejected,
        "eeg_channel_count": len(channel_names),
    }
