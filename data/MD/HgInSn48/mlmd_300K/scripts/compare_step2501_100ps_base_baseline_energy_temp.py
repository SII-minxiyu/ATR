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
    "finetuned_balanced": Path("runs/run_008_step2501_base_300K_100ps"),
    "baseline_random_e20": Path("runs/run_010_step2501_baseline_random_e20_300K_100ps"),
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


def max_force_norms(run_dir: Path) -> np.ndarray:
    values = []
    for atoms in read(run_dir / "mlmd_trajectory.extxyz", ":"):
        forces = atoms.get_forces()
        values.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(values, dtype=float)


def plot_one(series: dict[str, dict[str, np.ndarray]], key: str, ylabel: str, filename: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {
        "finetuned_balanced": "#1f77b4",
        "baseline_random_e20": "#d62728",
    }
    for label, data in series.items():
        ax.plot(data["time"], data[key], lw=1.1, label=label, color=colors.get(label))
    if key == "T_K":
        ax.axhline(300, color="#555555", lw=1.0, ls="--", alpha=0.75)
    ax.set_xlabel("Time since source_step=2501 start (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("runs/comparison_step2501_base_vs_baseline_100ps")
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    stats = []
    for label, run_dir in RUNS.items():
        log = parse_log(run_dir)
        natoms = len(read(run_dir / "final_structure.extxyz"))
        data = {
            "time": log[:, 0],
            "T_K": log[:, 4],
            "Epot_atom": log[:, 2] / natoms,
            "Ekin": log[:, 3],
        }
        fmax = max_force_norms(run_dir)
        if len(fmax) != len(log):
            raise RuntimeError(f"Fmax/log length mismatch for {run_dir}: {len(fmax)} vs {len(log)}")
        data["Fmax"] = fmax
        series[label] = data
        for key in ("T_K", "Epot_atom", "Ekin", "Fmax"):
            stats.append(stat_row(label, key, data[key]))

    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    plot_one(series, "T_K", "Temperature (K)", "temperature_finetuned_vs_baseline.png", out_dir)
    plot_one(series, "Epot_atom", "Epot/atom (eV)", "epot_atom_finetuned_vs_baseline.png", out_dir)
    plot_one(series, "Ekin", "Ekin (eV)", "ekin_finetuned_vs_baseline.png", out_dir)
    plot_one(series, "Fmax", "max |F| (eV/A)", "fmax_finetuned_vs_baseline.png", out_dir)
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
