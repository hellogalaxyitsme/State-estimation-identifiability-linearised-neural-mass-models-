from __future__ import annotations

import numpy as np

from neural_observability.working_memory.sources import (
    adjacent_bipolar_sources,
    bipolarize_trials,
    standard_1020_info,
)


def _electrode(name: str, x: float, label: str = "Hipp") -> dict[str, object]:
    return {
        "name": name,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "AnatomicalLocation": label,
    }


def test_adjacent_bipolar_sources_require_two_released_labels() -> None:
    names = ["A1", "A2", "A3", "B1", "B2"]
    rows = [
        _electrode("A1", 0.0),
        _electrode("A2", 1.0),
        _electrode("A3", 2.0, "no_label_found"),
        _electrode("B1", 4.0),
        _electrode("B2", 6.0),
    ]
    sources = adjacent_bipolar_sources(names, rows)
    assert [source.name for source in sources] == ["A1-A2", "B1-B2"]
    assert np.allclose(sources[0].midpoint_mni_mm, [0.5, 0.0, 0.0])
    assert np.allclose(sources[0].direction_mni, [1.0, 0.0, 0.0])


def test_bipolarize_is_lower_minus_higher() -> None:
    sources = adjacent_bipolar_sources(
        ["A1", "A2"], [_electrode("A1", 0), _electrode("A2", 1)]
    )
    trials = np.array([[[3.0, 5.0], [1.0, 2.0]]])
    assert np.allclose(bipolarize_trials(trials, sources), [[[2.0, 3.0]]])


def test_standard_montage_handles_legacy_temporal_names() -> None:
    info = standard_1020_info(["T1", "T2", "T3", "T4", "T5", "T6"], 200)
    assert info["nchan"] == 6
    assert all(np.all(np.isfinite(channel["loc"][:3])) for channel in info["chs"])


