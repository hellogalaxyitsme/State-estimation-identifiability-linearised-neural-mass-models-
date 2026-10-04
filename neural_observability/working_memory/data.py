"""Data access for the simultaneous scalp and intracranial working-memory recordings.

OpenNeuro ds004752 v1.0.1: dataset verification, repair of two EDF headers
whose start-time seconds field reads 60, pairing of scalp and intracranial
sessions and extraction of artefact-free trials.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neural_observability.statistics import sha256_file

ANNEX_KEY = re.compile(r"SHA256E-s(\d+)--([0-9a-f]{64})\.edf$")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the working-memory data configuration."""
    config_path = Path(path).expanduser().resolve()
    config = _read_json(config_path)
    required = {"dataset", "eligibility", "preprocessing", "head_model", "edf_header_repair", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    return config, config_path.parent.parent


def output_dir(config: dict[str, Any], project_root: Path) -> Path:
    path = Path(config["output_dir"])
    return path if path.is_absolute() else project_root / path


def _git_commit(dataset_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def annex_signal_files(dataset_root: Path) -> list[tuple[Path, int, str]]:
    """Return (link, size, sha256) for every annexed EDF of the dataset clone."""
    files = []
    for link in sorted(dataset_root.glob("sub-*/ses-*/*/*.edf")):
        match = ANNEX_KEY.search(os.readlink(link))
        if match is None:
            raise RuntimeError(f"Unrecognized annex key: {link}")
        files.append((link, int(match.group(1)), match.group(2)))
    return files


def verify_downloaded_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Check the dataset commit and every EDF against the dataset's own checksums."""
    root = Path(config["dataset"]["root"])
    if not root.is_dir() or _git_commit(root) != config["dataset"]["repository_commit"]:
        raise RuntimeError("The ds004752 clone is missing or at a different commit")
    files = annex_signal_files(root)
    if len(files) != int(config["dataset"]["signal_file_count"]):
        raise RuntimeError("The ds004752 signal-file count is wrong")
    total = 0
    for link, size, expected_sha in files:
        target = link.resolve(strict=True)
        if target.stat().st_size != size or sha256_file(target) != expected_sha:
            raise RuntimeError(f"Signal checksum failed: {link}")
        total += size
    if total != int(config["dataset"]["signal_total_size_bytes"]):
        raise RuntimeError("The total ds004752 signal size is wrong")
    return {"repository_commit": _git_commit(root), "signal_file_count": len(files), "signal_total_size_bytes": total}


def prepare_edf_header_repairs(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    """Copy the EDF files with an invalid start-time seconds field and correct that field.

    Only the two header bytes of the seconds field change; every data byte is
    identical to the released file, which is not modified.
    """
    repair = config["edf_header_repair"]
    root = Path(config["dataset"]["root"])
    expected = sorted(str(value) for value in repair["affected_relative_paths"])
    offset = int(repair["edf_header_second_byte_offset"])
    original_value = str(repair["invalid_second_value"]).encode("ascii")
    replacement = str(repair["replacement_second_value"]).encode("ascii")

    invalid: list[str] = []
    headers: dict[str, bytes] = {}
    for path in sorted(root.glob("sub-*/ses-*/*/*.edf")):
        relative = path.relative_to(root).as_posix()
        with path.open("rb") as stream:
            header = stream.read(256)
        if len(header) != 256:
            raise RuntimeError(f"Truncated EDF fixed header: {relative}")
        headers[relative] = header
        seconds = header[offset : offset + 2]
        if len(seconds) != 2 or not seconds.isdigit() or int(seconds) > 59:
            invalid.append(relative)
    if sorted(invalid) != expected:
        raise RuntimeError(f"EDF files with an invalid seconds field differ from the configuration: {invalid}")

    cache_root = Path(repair["repair_cache_root"])
    repairs: dict[str, Path] = {}
    rows = []
    for relative in expected:
        source = root / relative
        header = headers[relative]
        if header[offset : offset + 2] != original_value:
            raise RuntimeError(f"Unexpected EDF seconds field: {relative}")
        target = cache_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".building",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copyfile(source, temporary_path)
                with temporary_path.open("r+b") as stream:
                    stream.seek(offset)
                    if stream.read(2) != original_value:
                        raise RuntimeError("The copied EDF header changed before repair")
                    stream.seek(offset)
                    stream.write(replacement)
                if temporary_path.stat().st_size != source.stat().st_size:
                    raise RuntimeError("The EDF header repair changed file length")
                os.replace(temporary_path, target)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            target.chmod(0o444)
        with target.open("rb") as stream:
            repaired_header = stream.read(256)
        expected_header = header[:offset] + replacement + header[offset + 2 :]
        if repaired_header != expected_header or target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Repaired EDF copy is invalid: {relative}")
        repairs[str(Path(relative))] = target
        rows.append(
            {
                "relative_path": relative,
                "file_size_bytes": source.stat().st_size,
                "changed_byte_offsets": [offset, offset + 1],
                "original_ascii": original_value.decode("ascii"),
                "replacement_ascii": replacement.decode("ascii"),
            }
        )
    return repairs, {"repairs": rows}


def _paired_sessions(dataset_root: Path) -> list[tuple[str, str, Path, Path]]:
    sessions = []
    for eeg_dir in sorted(dataset_root.glob("sub-*/ses-*/eeg")):
        subject = eeg_dir.parts[-3]
        session = eeg_dir.parts[-2]
        ieeg_dir = eeg_dir.parent / "ieeg"
        if not ieeg_dir.is_dir():
            raise RuntimeError(f"Missing paired iEEG directory: {subject} {session}")
        sessions.append((subject, session, eeg_dir, ieeg_dir))
    return sessions


def _sidecar(eeg_or_ieeg_dir: Path, suffix: str) -> Path:
    matches = sorted(eeg_or_ieeg_dir.glob(f"*_{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix} below {eeg_or_ieeg_dir}")
    return matches[0]


def _validate_paired_events(
    eeg_rows: Sequence[dict[str, str]], ieeg_rows: Sequence[dict[str, str]]
) -> list[int]:
    if len(eeg_rows) != len(ieeg_rows):
        raise RuntimeError("Paired EEG/iEEG trial counts differ")
    shared = [
        "nTrial",
        "SetSize",
        "ProbeLetter",
        "Match",
        "Correct",
        "ResponseTime",
        "Artifact",
    ]
    keep = []
    for index, (eeg, ieeg) in enumerate(zip(eeg_rows, ieeg_rows)):
        if any(str(eeg[key]).strip() != str(ieeg[key]).strip() for key in shared):
            raise RuntimeError("Paired EEG/iEEG trial metadata differ")
        if int(eeg["Artifact"]) == 0:
            keep.append(index)
    return keep


def _extract_edf_trials(
    signal_path: Path,
    channel_names: Sequence[str],
    event_rows: Sequence[dict[str, str]],
    keep_indices: Sequence[int],
    expected_frequency: float,
) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_edf(signal_path, preload=True, verbose="ERROR")
    if not np.isclose(float(raw.info["sfreq"]), float(expected_frequency)):
        raise RuntimeError(f"EDF sampling frequency disagrees with its sidecar: {signal_path}")
    missing = sorted(set(channel_names) - set(raw.ch_names))
    if missing:
        raise RuntimeError(f"EDF channels are missing: {missing}")
    data = raw.get_data(picks=list(channel_names))
    trials = []
    expected_samples = int(round(8.0 * float(expected_frequency)))
    for index in keep_indices:
        row = event_rows[index]
        start = int(row["begSample"]) - 1
        stop = int(row["endSample"])
        if stop - start != expected_samples or start < 0 or stop > data.shape[1]:
            raise RuntimeError("An EDF trial does not span eight seconds within the recording")
        trials.append(data[:, start:stop])
    values = np.stack(trials, axis=0)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("EDF trials contain non-finite samples")
    return values


def _subject_seed(base_seed: int, subject: str, session: str, purpose: str) -> int:
    value = f"{base_seed}:{subject}:{session}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
