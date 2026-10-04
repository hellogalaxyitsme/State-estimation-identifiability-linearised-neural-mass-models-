"""Operating-point atlas: recoverability, mismatch floors and the finite-record forecast."""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from neural_observability.theory.atlas import analyse_point, sample_operating_points

CONFIG = {
    "seed": 20261001,
    "points_per_family": 200,
    "jitter_sd": 0.15,
    "min_damping": 0.5,
    "snr_db": [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
    "mismatch_levels": [0.05, 0.10, 0.20],
    "mismatch_draws": 8,
    "verification_snr_db": 10.0,
    "prior_log_sd": 0.2,
    "record_seconds": [0, 10, 60, 600, 3600, 36000],
    "verification_draws": 64,
    "high_snr_db": [60, 80, 100],
    "floor_check_snr_db": [40, 60, 80, 100],
    "second_drive_fractions": [0.0, 0.01, 0.1, 0.3],
    "second_drive_snr_db": [20, 40, 60, 80],
}


def _work(args):
    point, family, seed = args
    try:
        return analyse_point(point, CONFIG, seed, family)
    except Exception as exc:  # noqa: BLE001 - record failures, never drop silently
        return {"family": family, "error": repr(exc), "drive": point["drive"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="jansen_rit,wendling,wendling_2002,next_generation")
    parser.add_argument("--points", type=int, default=CONFIG["points_per_family"])
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--out", default="results/simulations/atlas")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG, points_per_family=args.points)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    family_index = {"jansen_rit": 0, "wendling": 1, "next_generation": 2, "wendling_2002": 3}
    inclusion_path = out / "inclusion.json"
    inclusion = json.loads(inclusion_path.read_text()) if inclusion_path.exists() else {}
    for family in args.families.split(","):
        f_index = family_index[family]
        start = time.time()
        counts: dict = {}
        points = sample_operating_points(
            family, args.points, config["seed"] + 1000 * f_index, config["jitter_sd"], config["min_damping"],
            record=counts,
        )
        inclusion[family] = counts
        (out / "inclusion.json").write_text(json.dumps(inclusion, indent=2))
        jobs = [(p, family, config["seed"] + 100000 * f_index + i) for i, p in enumerate(points)]
        with Pool(args.workers) as pool:
            records = pool.map(_work, jobs, chunksize=1)
        with open(out / f"{family}_records.jsonl", "w") as handle:
            for rec in records:
                handle.write(json.dumps(rec) + "\n")
        failures = sum("error" in r for r in records)
        print(f"{family}: {len(records)} points, {failures} failures, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
