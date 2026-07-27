#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read


def parse_times(log_path: Path) -> np.ndarray:
    rows = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append(float(line.split()[0]))
    if not rows:
        raise RuntimeError(f"No MD time rows found in {log_path}")
    return np.array(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_path = args.out if args.out is not None else run_dir / "fmax_timeseries.csv"

    times = parse_times(run_dir / "mlmd.log")
    frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
    if len(times) != len(frames):
        raise RuntimeError(f"Length mismatch: {len(times)} log rows vs {len(frames)} frames")

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "time_ps", "fmax_eV_per_A"])
        for idx, (time_ps, atoms) in enumerate(zip(times, frames)):
            forces = atoms.get_forces()
            fmax = float(np.linalg.norm(forces, axis=1).max())
            writer.writerow([idx, f"{time_ps:.8f}", f"{fmax:.10f}"])

    print(f"frames: {len(frames)}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
