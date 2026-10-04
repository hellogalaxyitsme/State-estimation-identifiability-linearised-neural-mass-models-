#!/usr/bin/env python3
"""Decide session eligibility and build the template-head forward model of each participant.

Uses channel, electrode and event metadata only; no signal is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_observability.working_memory.data import (
    _paired_sessions,
    _read_tsv,
    _sidecar,
    _validate_paired_events,
    load_config,
    output_dir,
)
from neural_observability.working_memory.sources import (
    adjacent_bipolar_sources,
    build_template_oriented_forward,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "working_memory_data.json")
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    dataset_root = Path(config["dataset"]["root"])
    output = output_dir(config, project_root)
    forward_root = output / "template_forward"
    forward_root.mkdir(parents=True, exist_ok=True)

    by_subject: dict[str, list[tuple[str, Path, Path]]] = {}
    for subject, session, eeg_dir, ieeg_dir in _paired_sessions(dataset_root):
        by_subject.setdefault(subject, []).append((session, eeg_dir, ieeg_dir))

    criteria = config["eligibility"]
    head = config["head_model"]
    subject_rows = []
    session_rows = []
    for number, subject in enumerate(sorted(by_subject), start=1):
        print(f"{number}/{len(by_subject)}: {subject}", flush=True)
        subject_sessions = sorted(by_subject[subject])
        _, first_eeg_dir, first_ieeg_dir = subject_sessions[0]
        eeg_names = [row["name"] for row in _read_tsv(_sidecar(first_eeg_dir, "channels.tsv"))]
        ieeg_names = [row["name"] for row in _read_tsv(_sidecar(first_ieeg_dir, "channels.tsv"))]
        sources = adjacent_bipolar_sources(ieeg_names, _read_tsv(_sidecar(first_ieeg_dir, "electrodes.tsv")))
        lead, retained, forward_detail = build_template_oriented_forward(
            eeg_names,
            sources,
            project_root / head["subjects_dir"],
            project_root / head["bem_solution"],
            float(config["preprocessing"]["target_sampling_frequency_hz"]),
        )
        np.savez_compressed(
            forward_root / f"{subject}_template_forward.npz",
            lead_field=lead,
            eeg_names=np.asarray(eeg_names),
            source_names=np.asarray([source.name for source in retained]),
            lower_names=np.asarray([source.lower_name for source in retained]),
            upper_names=np.asarray([source.upper_name for source in retained]),
            midpoint_mni_mm=np.asarray([source.midpoint_mni_mm for source in retained], dtype=float),
            direction_mni=np.asarray([source.direction_mni for source in retained], dtype=float),
        )
        session_passes = 0
        for session, eeg_dir, ieeg_dir in subject_sessions:
            failures = []
            current_eeg_names = [row["name"] for row in _read_tsv(_sidecar(eeg_dir, "channels.tsv"))]
            current_ieeg_names = [row["name"] for row in _read_tsv(_sidecar(ieeg_dir, "channels.tsv"))]
            current_sources = adjacent_bipolar_sources(
                current_ieeg_names, _read_tsv(_sidecar(ieeg_dir, "electrodes.tsv"))
            )
            if current_eeg_names != eeg_names:
                failures.append("eeg_channel_set_changed_within_subject")
            if [source.name for source in current_sources] != [source.name for source in sources]:
                failures.append("ieeg_source_set_changed_within_subject")
            try:
                clean = _validate_paired_events(
                    _read_tsv(_sidecar(eeg_dir, "events.tsv")), _read_tsv(_sidecar(ieeg_dir, "events.tsv"))
                )
            except RuntimeError as error:
                clean = []
                failures.append(f"paired_event_metadata_failure:{error}")
            if len(clean) < int(criteria["minimum_clean_trials_per_session"]):
                failures.append("too_few_clean_trials")
            if len(current_eeg_names) < int(criteria["minimum_scalp_channels"]):
                failures.append("too_few_scalp_channels")
            if len(current_ieeg_names) < int(criteria["minimum_ieeg_contacts"]):
                failures.append("too_few_ieeg_contacts")
            if len(retained) < int(criteria["minimum_valid_bipolar_sources"]):
                failures.append("too_few_template_forward_sources")
            session_passes += int(not failures)
            session_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "clean_trial_count": len(clean),
                    "scalp_channel_count": len(current_eeg_names),
                    "ieeg_contact_count": len(current_ieeg_names),
                    "anatomically_labelled_bipolar_source_count": len(current_sources),
                    "template_forward_source_count": len(retained),
                    "eligible": not failures,
                    "failure_reasons": ";".join(failures),
                }
            )
        eligible = session_passes > 0 and len(retained) >= int(criteria["minimum_valid_bipolar_sources"])
        subject_rows.append(
            {
                "subject": subject,
                "session_count": len(subject_sessions),
                "eligible_session_count": session_passes,
                "scalp_channel_count": len(eeg_names),
                "ieeg_contact_count": len(ieeg_names),
                "anatomically_labelled_bipolar_source_count": len(sources),
                "template_forward_source_count": len(retained),
                "template_forward_rejected_source_count": len(forward_detail["rejected_sources"]),
                "eligible": eligible,
                "failure_reasons": "" if eligible else "no_eligible_session_or_too_few_forward_sources",
            }
        )

    result = {
        "subject_count": len(subject_rows),
        "eligible_subject_count": sum(bool(row["eligible"]) for row in subject_rows),
        "subject_metadata": subject_rows,
        "session_metadata": session_rows,
    }
    path = output / "eligibility.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
