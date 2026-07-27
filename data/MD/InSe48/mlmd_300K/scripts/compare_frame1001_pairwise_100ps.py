#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


RUNS = {
    "finetuned_balanced": Path("runs/run_004_frame1001_300K_100ps"),
    "official_r2scan": Path("runs/run_005_official_r2scan_frame1001_300K_100ps"),
    "baseline_random_e20": Path("runs/run_006_baseline_random_e20_frame1001_300K_100ps"),
}
PAIRS = [
    ("finetuned_balanced", "official_r2scan", "comparison_frame1001_finetuned_vs_official_100ps"),
    ("finetuned_balanced", "baseline_random_e20", "comparison_frame1001_finetuned_vs_baseline_100ps"),
    ("official_r2scan", "baseline_random_e20", "comparison_frame1001_official_vs_baseline_100ps"),
]
COLORS = {
    "finetuned_balanced": "#1f77b4",
    "official_r2scan": "#2ca02c",
    "baseline_random_e20": "#d62728",
}


def parse_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"No MD rows found in {run_dir / 'mlmd.log'}")
    return np.array(rows, dtype=float)


def dmin_series(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        dist = atoms.get_all_distances(mic=True)
        vals.append(float(dist[np.triu_indices(len(atoms), 1)].min()))
    return np.array(vals, dtype=float)


def fmax_series(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        forces = atoms.get_forces()
        vals.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(vals, dtype=float)


def stat_row(label: str, quantity: str, values: np.ndarray) -> dict[str, str]:
    q5, q50, q95 = np.percentile(values, [5, 50, 95])
    return {
        "label": label,
        "quantity": quantity,
        "mean": f"{values.mean():.8g}",
        "std": f"{values.std():.8g}",
        "min": f"{values.min():.8g}",
        "p5": f"{q5:.8g}",
        "median": f"{q50:.8g}",
        "p95": f"{q95:.8g}",
        "max": f"{values.max():.8g}",
    }


def load_all() -> dict[str, dict[str, np.ndarray]]:
    series = {}
    for label, run_dir in RUNS.items():
        log = parse_log(run_dir)
        frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
        natoms = len(frames[0])
        if len(log) != len(frames):
            raise RuntimeError(f"Length mismatch for {label}: {len(log)} log rows vs {len(frames)} frames")
        series[label] = {
            "time": log[:, 0],
            "T_K": log[:, 4],
            "Epot_atom": log[:, 2] / natoms,
            "Ekin": log[:, 3],
            "Fmax": fmax_series(frames),
            "dmin": dmin_series(frames),
        }
    return series


def plot_pair(pair_series: dict[str, dict[str, np.ndarray]], key: str, ylabel: str, filename: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, data in pair_series.items():
        ax.plot(data["time"], data[key], lw=1.0, label=label, color=COLORS[label])
    if key == "T_K":
        ax.axhline(300, color="#555555", lw=1.0, ls="--", alpha=0.75)
    ax.set_xlabel("Time since frame_1001 start (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def write_pair(label_a: str, label_b: str, out_name: str, series: dict[str, dict[str, np.ndarray]]) -> None:
    out_dir = Path("runs") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_series = {label_a: series[label_a], label_b: series[label_b]}

    stats = []
    for label, data in pair_series.items():
        for key in ("T_K", "Epot_atom", "Ekin", "Fmax", "dmin"):
            stats.append(stat_row(label, key, data[key]))
    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    plot_pair(pair_series, "T_K", "Temperature (K)", "temperature_compare_100ps.png", out_dir)
    plot_pair(pair_series, "Epot_atom", "Epot/atom (eV)", "epot_atom_compare_100ps.png", out_dir)
    plot_pair(pair_series, "Ekin", "Ekin (eV)", "ekin_compare_100ps.png", out_dir)
    plot_pair(pair_series, "Fmax", "max |F| (eV/A)", "fmax_compare_100ps.png", out_dir)
    plot_pair(pair_series, "dmin", "Minimum pair distance (A)", "dmin_compare_100ps.png", out_dir)
    print(f"wrote: {out_dir}")


def main() -> None:
    series = load_all()
    for label_a, label_b, out_name in PAIRS:
        write_pair(label_a, label_b, out_name, series)


if __name__ == "__main__":
    main()
