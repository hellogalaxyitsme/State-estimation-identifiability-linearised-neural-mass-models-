from __future__ import annotations

from pathlib import Path

from neural_observability.working_memory.data import (
    _validate_paired_events,
    load_config,
    prepare_edf_header_repairs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_names_the_analysed_dataset_commit() -> None:
    config, root = load_config(ROOT / "configs" / "working_memory_data.json")
    assert root == ROOT
    assert config["dataset"]["repository_commit"] == "234fd4c04dea66ab9b0fd5ddbf4ec51f56b6ad02"
    assert len(config["edf_header_repair"]["affected_relative_paths"]) == 2


def test_paired_event_validation_uses_only_released_artifact_flag() -> None:
    rows = []
    for number, artifact in [(1, "0"), (2, "1"), (3, "0")]:
        rows.append(
            {
                "nTrial": str(number),
                "SetSize": "4",
                "ProbeLetter": "A",
                "Match": "IN",
                "Correct": "1",
                "ResponseTime": "1.0",
                "Artifact": artifact,
            }
        )
    assert _validate_paired_events(rows, [dict(row) for row in rows]) == [0, 2]
    changed = [dict(row) for row in rows]
    changed[0]["SetSize"] = "8"
    try:
        _validate_paired_events(rows, changed)
    except RuntimeError as error:
        assert "metadata differ" in str(error)
    else:
        raise AssertionError("Mismatched modality metadata must fail")


def test_edf_header_repair_changes_only_the_seconds_field(tmp_path) -> None:
    root = tmp_path / "dataset"
    relative = "sub-01/ses-01/eeg/sub-01_ses-01_task-test_eeg.edf"
    source = root / relative
    source.parent.mkdir(parents=True)
    header = bytearray(b" " * 256)
    header[168:176] = b"18.08.26"
    header[176:184] = b"12.34.60"
    payload = bytes(range(128))
    source.write_bytes(bytes(header) + payload)
    config = {
        "dataset": {"root": str(root)},
        "edf_header_repair": {
            "affected_relative_paths": [relative],
            "edf_header_second_byte_offset": 182,
            "invalid_second_value": "60",
            "replacement_second_value": "59",
            "repair_cache_root": str(tmp_path / "repairs"),
        },
    }

    repairs, manifest = prepare_edf_header_repairs(config)
    repaired = repairs[str(Path(relative))].read_bytes()
    original = source.read_bytes()
    assert original[182:184] == b"60"
    assert repaired[182:184] == b"59"
    assert repaired[:182] == original[:182]
    assert repaired[184:] == original[184:]
    assert manifest["repairs"][0]["changed_byte_offsets"] == [182, 183]
