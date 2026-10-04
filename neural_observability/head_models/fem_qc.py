"""Segmentation and electrode-projection overlays and summaries for head-model QC."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def make_segmentation_overlay(
    t1_path: str | Path, labels_path: str | Path, output_path: str | Path, subject: str
) -> Path:
    """Render nine canonical slices with categorical tissue boundaries."""
    t1 = nib.as_closest_canonical(nib.load(t1_path))
    labels = nib.as_closest_canonical(nib.load(labels_path))
    if len(t1.shape) != 3:
        raise ValueError(f"The T1 image must be 3-D, got shape {t1.shape}")
    # SimNIBS 4.6 writes final_tissues as X x Y x Z x 1. Selecting the sole
    # volume preserves its affine while avoiding nibabel's 4-D resampling map.
    if len(labels.shape) == 4 and labels.shape[3] == 1:
        labels = labels.slicer[..., 0]
    if len(labels.shape) != 3:
        raise ValueError(f"The tissue label image must be 3-D, got shape {labels.shape}")
    if labels.shape != t1.shape or not np.allclose(labels.affine, t1.affine):
        labels = resample_from_to(labels, (t1.shape, t1.affine), order=0)
    anatomy = np.asarray(t1.dataobj, dtype=float)
    tissue = np.rint(np.asarray(labels.dataobj, dtype=float)).astype(int)
    finite = anatomy[np.isfinite(anatomy)]
    if finite.size == 0:
        raise ValueError("The T1 image has no finite intensities")
    lower, upper = np.quantile(finite, [0.01, 0.995])
    if upper <= lower:
        raise ValueError("The T1 image has degenerate intensity range")
    anatomy = np.clip((anatomy - lower) / (upper - lower), 0.0, 1.0)
    edges = ndimage.maximum_filter(tissue, size=3) != ndimage.minimum_filter(
        tissue, size=3
    )
    planes = [(0, "sagittal"), (1, "coronal"), (2, "axial")]
    fractions = [0.25, 0.50, 0.75]
    figure, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    cmap = plt.get_cmap("turbo", max(10, int(tissue.max()) + 1))
    for row, (axis, plane_name) in enumerate(planes):
        for column, fraction in enumerate(fractions):
            index = int(round(fraction * (t1.shape[axis] - 1)))
            background = np.rot90(np.take(anatomy, index, axis=axis))
            label_slice = np.rot90(np.take(tissue, index, axis=axis))
            edge_slice = np.rot90(np.take(edges, index, axis=axis))
            overlay = np.ma.masked_where(~edge_slice, label_slice)
            axes[row, column].imshow(background, cmap="gray", vmin=0, vmax=1)
            axes[row, column].imshow(
                overlay,
                cmap=cmap,
                vmin=0,
                vmax=max(9, int(tissue.max())),
                alpha=0.9,
                interpolation="nearest",
            )
            axes[row, column].set_title(
                f"{plane_name} {fraction:.0%} (index {index})"
            )
            axes[row, column].axis("off")
    figure.suptitle(
        f"{subject}: CHARM tissue boundaries on canonical T1\n"
        "Review registration, brain/skull/scalp continuity, holes, and gross leakage",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def make_electrode_projection_overlay(
    scalp_vertices_mm: np.ndarray,
    raw_positions_mm: np.ndarray,
    projected_positions_mm: np.ndarray,
    raw_landmarks_mm: np.ndarray,
    projected_landmarks_mm: np.ndarray,
    landmark_names: list[str],
    output_path: str | Path,
    subject: str,
) -> Path:
    """Render orthographic cap projection, displacement, and fiducial QC."""
    scalp = np.asarray(scalp_vertices_mm, dtype=float)
    raw = np.asarray(raw_positions_mm, dtype=float)
    projected = np.asarray(projected_positions_mm, dtype=float)
    raw_landmarks = np.asarray(raw_landmarks_mm, dtype=float)
    projected_landmarks = np.asarray(projected_landmarks_mm, dtype=float)
    if scalp.ndim != 2 or scalp.shape[1] != 3 or len(scalp) < 4:
        raise ValueError("scalp_vertices_mm must be an N x 3 matrix")
    if raw.ndim != 2 or raw.shape[1] != 3 or projected.shape != raw.shape:
        raise ValueError("Raw/projected electrode coordinates are incompatible")
    if raw_landmarks.shape != (len(landmark_names), 3) or (
        projected_landmarks.shape != raw_landmarks.shape
    ):
        raise ValueError("Landmark names and coordinates are incompatible")
    arrays = [scalp, raw, projected, raw_landmarks, projected_landmarks]
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Projection overlay coordinates must be finite")

    projection_distances = np.linalg.norm(projected - raw, axis=1)
    planes = [((0, 1), "axial (x, y)"), ((0, 2), "coronal (x, z)"), ((1, 2), "sagittal (y, z)")]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    colored = None
    for axis, (dimensions, title) in zip(axes, planes):
        first, second = dimensions
        axis.scatter(
            scalp[:, first],
            scalp[:, second],
            s=1,
            c="#777777",
            alpha=0.18,
            linewidths=0,
            rasterized=True,
            label="scalp surface",
        )
        segments = np.stack(
            [raw[:, [first, second]], projected[:, [first, second]]], axis=1
        )
        axis.add_collection(
            LineCollection(segments, colors="#f28e2b", linewidths=0.7, alpha=0.65)
        )
        colored = axis.scatter(
            raw[:, first],
            raw[:, second],
            c=projection_distances,
            cmap="magma",
            s=16,
            edgecolors="none",
            label="released electrode",
            zorder=3,
        )
        axis.scatter(
            projected[:, first],
            projected[:, second],
            facecolors="none",
            edgecolors="#1f77b4",
            s=24,
            linewidths=0.8,
            label="scalp projection",
            zorder=4,
        )
        for index, name in enumerate(landmark_names):
            axis.plot(
                [raw_landmarks[index, first], projected_landmarks[index, first]],
                [raw_landmarks[index, second], projected_landmarks[index, second]],
                color="#2ca02c",
                linewidth=1.4,
                zorder=5,
            )
            axis.scatter(
                raw_landmarks[index, first],
                raw_landmarks[index, second],
                marker="x",
                c="#d62728",
                s=55,
                linewidths=1.4,
                zorder=6,
            )
            axis.annotate(
                name,
                (raw_landmarks[index, first], raw_landmarks[index, second]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                zorder=7,
            )
        axis.set_title(title)
        axis.set_xlabel(f"{('x', 'y', 'z')[first]} (mm)")
        axis.set_ylabel(f"{('x', 'y', 'z')[second]} (mm)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes[0].legend(loc="best", fontsize=8)
    if colored is not None:
        colorbar = figure.colorbar(colored, ax=axes, shrink=0.72, pad=0.02)
        colorbar.set_label("released-to-scalp projection distance (mm)")
    figure.suptitle(
        f"{subject}: released cap (filled) projected to CHARM scalp (rings)\n"
        "Review fiducials, lateral displacement, cap topology, mirroring, and gross scale/translation",
        fontsize=13,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def write_qc_cohort_summary(qc_dir: str | Path) -> tuple[Path, Path]:
    """Write a flat metric table and a cohort QC summary."""
    directory = Path(qc_dir)
    rows = []
    failures: Counter[str] = Counter()
    for path in sorted(directory.glob("sub-*_automated_qc.json")):
        qc = json.loads(path.read_text(encoding="utf-8"))
        electrode = qc["electrode_projection_qc"]
        mesh = qc["tetrahedral_numerical_quality"]
        source = qc["source_to_brain_conservative_node_distance_mm"]
        failed = sorted(name for name, passed in qc["checks"].items() if not passed)
        failures.update(failed)
        rows.append(
            {
                "subject": qc["subject"],
                "automated_qc_pass": qc["automated_qc_pass"],
                "failed_checks": ";".join(failed),
                "tetrahedron_count": qc["tetrahedron_count"],
                "invalid_tetrahedron_count": qc["invalid_tetrahedron_count"],
                "mean_ratio_p001": mesh["mean_ratio"]["p001"],
                "fraction_mean_ratio_below_0_01": mesh[
                    "fraction_mean_ratio_below_0_01"
                ],
                "minimum_dihedral_p001_degrees": mesh[
                    "minimum_dihedral_angle_degrees"
                ]["p001"],
                "maximum_dihedral_p999_degrees": mesh[
                    "maximum_dihedral_angle_degrees"
                ]["p999"],
                "edge_aspect_ratio_p999": mesh["edge_aspect_ratio"]["p999"],
                "jacobian_condition_number_p999": mesh[
                    "regular_reference_jacobian_condition_number"
                ]["p999"],
                "projection_distance_median_mm": electrode["projection_distance_mm"][
                    "median"
                ],
                "projection_distance_p95_mm": electrode["projection_distance_mm"][
                    "p95"
                ],
                "projection_distance_maximum_mm": electrode[
                    "projection_distance_mm"
                ]["maximum"],
                "signed_outward_projection_fraction_descriptive": electrode[
                    "outward_projection_fraction"
                ],
                "radially_aligned_projection_fraction": electrode[
                    "radially_aligned_projection_fraction"
                ],
                "radial_alignment_absolute_cosine_median": electrode[
                    "radial_alignment_absolute_cosine"
                ]["median"],
                "angular_displacement_p95_degrees": electrode[
                    "angular_displacement_degrees"
                ]["p95"],
                "neighbour_overlap_fraction_mean": electrode[
                    "neighbour_overlap_fraction"
                ]["mean"],
                "minimum_projected_interelectrode_distance_mm": electrode[
                    "minimum_projected_interelectrode_distance_mm"
                ],
                "maximum_fiducial_to_scalp_distance_mm": electrode[
                    "maximum_fiducial_to_scalp_distance_mm"
                ],
                "centroid_projection_displacement_mm": electrode[
                    "centroid_projection_displacement_mm"
                ],
                "radial_scale_ratio_raw_over_projected_median": electrode[
                    "radial_scale_ratio_raw_over_projected"
                ]["median"],
                "source_to_brain_distance_median_mm": source["median"],
                "source_to_brain_distance_p95_mm": source["p95"],
                "source_to_brain_distance_maximum_mm": source["maximum"],
            }
        )
    if not rows:
        raise ValueError(f"No automated QC records found in {directory}")
    csv_path = directory / "headmodel_qc_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "subject_count": len(rows),
        "automated_pass_count": sum(row["automated_qc_pass"] is True for row in rows),
        "automated_fail_count": sum(row["automated_qc_pass"] is False for row in rows),
        "failed_subjects": [
            row["subject"] for row in rows if row["automated_qc_pass"] is False
        ],
        "failure_count_by_check": dict(sorted(failures.items())),
        "metric_table": csv_path.name,
    }
    json_path = directory / "headmodel_qc_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def make_qc_review_index(qc_dir: str | Path) -> Path:
    """Build an HTML index of the QC overlays and reports for visual review."""
    directory = Path(qc_dir)
    sections = []
    for qc_path in sorted(directory.glob("sub-*_automated_qc.json")):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        subject = str(qc["subject"])
        details = {
            "Tissue tetrahedra": qc.get("tissue_tetrahedron_counts", {}),
            "EEG-to-scalp mm": qc.get("eeg_to_scalp_distance_mm", {}),
            "Electrode projection": qc.get("electrode_projection_qc", {}),
            "Tetrahedral quality": qc.get("tetrahedral_numerical_quality", {}),
            "Source-to-brain mm": qc.get("source_to_brain_conservative_node_distance_mm", {}),
        }
        rows = "<br>\n     ".join(
            f"{name}: {html.escape(json.dumps(value, sort_keys=True))}"
            for name, value in details.items()
        )
        sections.append(
            f"""
<section id="{html.escape(subject)}">
  <h2>{html.escape(subject)}</h2>
  <p>Automated pass: <strong>{qc.get("automated_qc_pass") is True}</strong><br>
     {rows}</p>
  <p><a href="{html.escape(subject)}_charm_report.html">CHARM report</a></p>
  <img src="{html.escape(subject)}_segmentation_overlay.png" alt="{html.escape(subject)} segmentation overlay">
  <img src="{html.escape(subject)}_electrode_projection_overlay.png" alt="{html.escape(subject)} electrode projection overlay">
</section>"""
        )
    body = "\n".join(sections) or "<p>No QC records found.</p>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Head-model visual QC</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: auto; line-height: 1.4; }}
section {{ border-top: 2px solid #444; padding: 1rem 0 2rem; }}
img {{ width: 100%; height: auto; border: 1px solid #aaa; }}
</style></head><body>
<h1>Head-model visual QC</h1>
<p>Review each segmentation overlay, electrode-projection overlay and CHARM
report, and record the decision in <code>visual_qc.csv</code> (columns
<code>subject</code>, <code>visual_qc_status</code>).</p>
{body}
</body></html>
"""
    output = directory / "visual_qc_review_index.html"
    output.write_text(document, encoding="utf-8")
    return output
