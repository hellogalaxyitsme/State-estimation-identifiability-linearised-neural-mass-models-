"""Jansen--Rit equilibria and coupled-network linearizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq, root
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class JRParameters:
    """Standard Jansen--Rit parameters in mV and seconds."""

    gain_e: float = 3.25
    gain_i: float = 22.0
    a: float = 100.0
    b: float = 50.0
    c1: float = 135.0
    c2: float = 108.0
    c3: float = 33.75
    c4: float = 33.75
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def sigmoid(value: np.ndarray | float, parameters: JRParameters) -> np.ndarray:
    """Jansen--Rit firing-rate sigmoid."""
    value_array = np.asarray(value, dtype=float)
    exponent = np.clip(parameters.r * (parameters.v0 - value_array), -700, 700)
    return 2.0 * parameters.e0 / (1.0 + np.exp(exponent))


def sigmoid_prime(value: np.ndarray | float, parameters: JRParameters) -> np.ndarray:
    """Derivative of the firing-rate sigmoid."""
    rate = sigmoid(value, parameters)
    return parameters.r * rate * (1.0 - rate / (2.0 * parameters.e0))


def single_column_equilibria(
    p0: float, parameters: JRParameters = JRParameters()
) -> list[np.ndarray]:
    """Return all one-column equilibria ordered by the pyramidal state x1."""
    p = parameters

    def remaining_states(x1: float) -> tuple[float, float]:
        x2 = p.gain_e / p.a * (p0 + p.c2 * sigmoid(p.c1 * x1, p))
        x3 = p.gain_i / p.b * p.c4 * sigmoid(p.c3 * x1, p)
        return float(x2), float(x3)

    def residual(x1: float) -> float:
        x2, x3 = remaining_states(x1)
        return float(x1 - p.gain_e / p.a * sigmoid(x2 - x3, p))

    maximum_x1 = p.gain_e / p.a * 2.0 * p.e0
    grid = np.linspace(0.0, maximum_x1, 10_001)
    roots: list[float] = []
    grid_x2 = p.gain_e / p.a * (p0 + p.c2 * sigmoid(p.c1 * grid, p))
    grid_x3 = p.gain_i / p.b * p.c4 * sigmoid(p.c3 * grid, p)
    values = grid - p.gain_e / p.a * sigmoid(grid_x2 - grid_x3, p)
    for lower, upper, f_lower, f_upper in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if f_lower == 0.0 or f_lower * f_upper < 0.0:
            solution = brentq(residual, lower, upper)
            if not roots or abs(solution - roots[-1]) > 1e-9:
                roots.append(solution)
    if not roots:
        raise RuntimeError(f"No Jansen--Rit equilibrium found for p0={p0}")
    return [np.array([x1, *remaining_states(x1)]) for x1 in roots]


def select_equilibrium(
    p0: float, branch: str, parameters: JRParameters = JRParameters()
) -> np.ndarray:
    """Select the lowest or highest one-column equilibrium branch."""
    equilibria = single_column_equilibria(p0, parameters)
    if branch == "lowest":
        return equilibria[0]
    if branch == "highest":
        return equilibria[-1]
    raise ValueError("branch must be 'lowest' or 'highest'")


def local_jacobian(
    equilibrium: np.ndarray, parameters: JRParameters = JRParameters()
) -> np.ndarray:
    """Return the corrected six-state Jansen--Rit Jacobian."""
    p = parameters
    x1, x2, x3 = np.asarray(equilibrium, dtype=float)
    k_p = p.gain_e * p.a * sigmoid_prime(x2 - x3, p)
    k_e = p.gain_e * p.a * p.c1 * p.c2 * sigmoid_prime(p.c1 * x1, p)
    k_i = p.gain_i * p.b * p.c3 * p.c4 * sigmoid_prime(p.c3 * x1, p)
    return np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [-p.a**2, k_p, -k_p, -2 * p.a, 0, 0],
            [k_e, -p.a**2, 0, 0, -2 * p.a, 0],
            [k_i, 0, -p.b**2, 0, 0, -2 * p.b],
        ],
        dtype=float,
    )


def spatial_graph(
    positions: np.ndarray, nearest_neighbors: int, length_scale_m: float
) -> np.ndarray:
    """Build a row-normalized distance graph from source/parcel centroids."""
    positions = np.asarray(positions, dtype=float)
    source_count = len(positions)
    if source_count < 2:
        return np.zeros((source_count, source_count), dtype=float)
    neighbor_count = min(max(1, nearest_neighbors), source_count - 1)
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, np.inf)
    graph = np.zeros_like(distances)
    for row in range(source_count):
        neighbors = np.argpartition(distances[row], neighbor_count)[:neighbor_count]
        graph[row, neighbors] = np.exp(
            -0.5 * (distances[row, neighbors] / length_scale_m) ** 2
        )
    row_sums = graph.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise RuntimeError("Spatial graph contains an isolated node")
    return graph / row_sums


def _network_equilibrium(
    drives: np.ndarray,
    graph: np.ndarray,
    coupling_gain: float,
    branch: str,
    parameters: JRParameters,
) -> tuple[np.ndarray, float]:
    """Solve the coupled network equilibrium in x1 and recover x2/x3."""
    p = parameters
    initial_x1 = np.array(
        [select_equilibrium(float(drive), branch, p)[0] for drive in drives]
    )

    def states_from_x1(x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        firing = sigmoid(p.c1 * x1, p)
        coupled_input = coupling_gain * (graph @ firing)
        x2 = p.gain_e / p.a * (drives + p.c2 * firing + coupled_input)
        x3 = p.gain_i / p.b * p.c4 * sigmoid(p.c3 * x1, p)
        return x2, x3

    def residual(x1: np.ndarray) -> np.ndarray:
        x2, x3 = states_from_x1(x1)
        return x1 - p.gain_e / p.a * sigmoid(x2 - x3, p)

    solution = root(residual, initial_x1, method="hybr", options={"xtol": 1e-10})
    residual_norm = float(np.linalg.norm(residual(solution.x), ord=np.inf))
    if not solution.success or residual_norm > 1e-7:
        raise RuntimeError(
            "Coupled Jansen--Rit equilibrium solve failed: "
            f"{solution.message}; residual={residual_norm:.3e}"
        )
    x2, x3 = states_from_x1(solution.x)
    return np.column_stack([solution.x, x2, x3]), residual_norm


def build_network_linearization(
    positions: np.ndarray,
    scenario: dict[str, Any],
    graph_config: dict[str, Any],
    rng: np.random.Generator,
    parameters: JRParameters = JRParameters(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a model-consistent coupled Jansen--Rit network Jacobian.

    Returns ``(A, H, equilibrium, metadata)``. ``H`` maps the full state to
    one pyramidal-potential difference per region.
    """
    positions = np.asarray(positions, dtype=float)
    region_count = len(positions)
    graph = spatial_graph(
        positions,
        int(graph_config["nearest_neighbors"]),
        float(graph_config["length_scale_m"]),
    )
    relative_sd = float(scenario["drive_relative_sd"])
    p0 = float(scenario["p0_per_second"])
    if relative_sd:
        drives = p0 * (1.0 + relative_sd * rng.standard_normal(region_count))
        drives = np.maximum(drives, 1e-6)
    else:
        drives = np.full(region_count, p0)
    coupling_gain = float(scenario["coupling_gain"])
    equilibrium, residual_norm = _network_equilibrium(
        drives,
        graph,
        coupling_gain,
        str(scenario["equilibrium_branch"]),
        parameters,
    )

    state_count = 6 * region_count
    network_a = np.zeros((state_count, state_count), dtype=float)
    local_h = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    source_h = np.zeros((region_count, state_count), dtype=float)
    for region in range(region_count):
        block = slice(6 * region, 6 * region + 6)
        network_a[block, block] = local_jacobian(equilibrium[region], parameters)
        source_h[region, block] = local_h

    # External coupling enters the excitatory PSP equation through firing from
    # the presynaptic pyramidal population S(C1*x1_j).
    firing_slopes = parameters.c1 * sigmoid_prime(
        parameters.c1 * equilibrium[:, 0], parameters
    )
    coupling_derivative = (
        parameters.gain_e
        * parameters.a
        * coupling_gain
        * graph
        * firing_slopes[np.newaxis, :]
    )
    for target in range(region_count):
        row = 6 * target + 4
        for source in np.flatnonzero(coupling_derivative[target]):
            network_a[row, 6 * source] += coupling_derivative[target, source]

    spectral_abscissa = float(np.max(np.linalg.eigvals(network_a).real))
    metadata: dict[str, Any] = {
        "name": scenario["name"],
        "region_count": region_count,
        "state_count": state_count,
        "drive_min_per_second": float(drives.min()),
        "drive_mean_per_second": float(drives.mean()),
        "drive_max_per_second": float(drives.max()),
        "coupling_gain": coupling_gain,
        "equilibrium_branch": scenario["equilibrium_branch"],
        "equilibrium_residual_inf": residual_norm,
        "spectral_abscissa_per_second": spectral_abscissa,
    }
    return network_a, source_h, equilibrium, metadata
