"""Dependency-light ccepcoreg stimulation-label and geometry utilities."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


PAIR_RE = re.compile(r"^(?P<shaft>.*?)(?P<a>\d+)-(?P<b>\d+)$")
DESCRIPTION_RE = re.compile(
    r"^Stimulation of (?P<pair>.+?)\s+"
    # One released sidecar contains the transparent typo ``5msma``.
    r"(?P<intensity>[0-9.]+)(?:ms)?ma\s+(?P<width>[0-9.]+)ms\b",
    flags=re.IGNORECASE,
)


def parse_stimulation_description(description: str) -> dict[str, Any]:
    """Parse the stimulation-site, current, and pulse-width fields."""
    match = DESCRIPTION_RE.match(description.strip())
    if match is None:
        raise ValueError(f"Cannot parse stimulation description: {description!r}")
    return {
        "source_label": match.group("pair"),
        "intensity_ma": float(match.group("intensity")),
        "pulse_width_ms": float(match.group("width")),
    }


def infer_bipolar_orientation(
    source_label: str, positions: dict[str, np.ndarray]
) -> tuple[np.ndarray, str]:
    """Infer lower-to-higher-contact shaft direction from bipolar midpoints."""
    match = PAIR_RE.match(source_label)
    if match is None:
        raise ValueError(f"Cannot parse bipolar contact label: {source_label!r}")
    shaft = match.group("shaft")
    a = int(match.group("a"))
    b = int(match.group("b"))
    if b != a + 1:
        raise ValueError(f"Expected adjacent contacts in {source_label!r}")
    previous = f"{shaft}{a - 1}-{a}"
    following = f"{shaft}{b}-{b + 1}"
    if previous in positions and following in positions:
        vector = positions[following] - positions[previous]
        method = "central_adjacent_pairs"
    elif following in positions:
        vector = positions[following] - positions[source_label]
        method = "forward_adjacent_pair"
    elif previous in positions:
        vector = positions[source_label] - positions[previous]
        method = "backward_adjacent_pair"
    else:
        raise ValueError(f"No adjacent pair can orient {source_label!r}")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Degenerate orientation for {source_label!r}")
    return vector / norm, method

