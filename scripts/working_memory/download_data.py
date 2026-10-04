#!/usr/bin/env python3
"""Clone OpenNeuro ds004752 at the analysed commit and download its EDF recordings.

Each recording is downloaded from the OpenNeuro S3 store and checked against
the size and SHA-256 checksum recorded in the dataset's git-annex key.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path

from neural_observability.statistics import sha256_file
from neural_observability.working_memory.data import annex_signal_files, load_config

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "working_memory_data.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    dataset = config["dataset"]
    root = Path(dataset["root"])
    if not root.exists():
        subprocess.run(["git", "clone", dataset["repository_url"], str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", dataset["repository_commit"]], check=True)

    jobs = []
    for link, size, expected_sha in annex_signal_files(root):
        target = (link.parent / os.readlink(link)).resolve(strict=False)
        url = f"{dataset['signal_base_url']}/{link.relative_to(root).as_posix()}"
        jobs.append((link, target, size, expected_sha, url))

    def download(job: tuple[Path, Path, int, str, str]) -> str:
        link, target, size, expected_sha, url = job
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            part = target.with_name(f"{target.name}.part")
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                 "--continue-at", "-", "--output", str(part), url],
                check=True,
            )
            os.replace(part, target)
        if target.stat().st_size != size or sha256_file(target) != expected_sha:
            raise RuntimeError(f"Checksum mismatch: {link}")
        return str(link.relative_to(root))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, path in enumerate(pool.map(download, jobs), start=1):
            print(f"{number}/{len(jobs)} {path}", flush=True)


if __name__ == "__main__":
    main()
