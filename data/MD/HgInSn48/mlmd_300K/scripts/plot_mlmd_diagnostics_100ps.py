#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


COLORS = {"Hg": "#4d4d4d", "In": "#2f80ed", "Sn": "#d95f02"}


def parse_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"No MD rows found in {run_dir / 'mlmd.log'}")
    return np.array(rows, dtype=float)


def min_distances(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        d = atoms.get_all_distances(mic=True)
        vals.append(float(d[np.triu_indices(len(atoms), 1)].min()))
    return np.array(vals, dtype=float)


def max_force_norms(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        forces = atoms.get_forces()
        vals.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(vals, dtype=float)


def stat_row(quantity: str, values: np.ndarray) -> dict[str, str]:
    values = values[np.isfinite(values)]
    q5, q50, q95 = np.percentile(values, [5, 50, 95])
    return {
        "quantity": quantity,
        "mean": f"{values.mean():.8g}",
        "std": f"{values.std():.8g}",
        "min": f"{values.min():.8g}",
        "p5": f"{q5:.8g}",
        "median": f"{q50:.8g}",
        "p95": f"{q95:.8g}",
        "max": f"{values.max():.8g}",
    }


def plot_temperature_energy(data: np.ndarray, out_path: Path) -> None:
    t = data[:, 0]
    etot = data[:, 1]
    epot = data[:, 2]
    temp = data[:, 4]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(t, temp, color="#1f77b4", lw=1.2)
    axes[0].axhline(300, color="#555555", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("ML-MD 300 K NVT diagnostics")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, epot, label="Epot", color="#d62728", lw=1.1)
    axes[1].plot(t, etot, label="Etot", color="#2ca02c", lw=1.0)
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].legend(frameon=False, ncols=2)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_potential_kinetic_force(data: np.ndarray, max_forces: np.ndarray, out_path: Path) -> None:
    t = data[:, 0]
    epot = data[:, 2]
    ekin = data[:, 3]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
    axes[0].plot(t, epot, color="#d62728", lw=1.1)
    axes[0].set_ylabel("Epot (eV)")
    axes[0].set_title("Potential energy, kinetic energy, and maximum force")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, ekin, color="#ff7f0e", lw=1.1)
    axes[1].set_ylabel("Ekin (eV)")
    axes[1].grid(alpha=0.25)
    axes[2].plot(t, max_forces, color="#6a3d9a", lw=1.0)
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel("max |F| (eV/A)")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_min_distance(times: np.ndarray, mins: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(times, mins, color="#6a3d9a", lw=1.0)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Minimum pair distance (A)")
    ax.set_title("Closest atom pair over trajectory")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_snapshots(frames, data: np.ndarray, out_path: Path) -> None:
    sample_ps = [0.0, 50.0, 100.0]
    times = data[:, 0]
    indices = [int(np.argmin(np.abs(times - t))) for t in sample_ps]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for col, frame_idx in enumerate(indices):
        atoms = frames[frame_idx]
        symbols = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = atoms.cell.lengths()
        title = f"{times[frame_idx]:.1f} ps"
        for row, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[row, col]
            for element in ("Hg", "In", "Sn"):
                mask = symbols == element
                ax.scatter(
                    pos[mask, dims[0]],
                    pos[mask, dims[1]],
                    s=42,
                    c=COLORS[element],
                    label=element if row == 0 and col == 0 else None,
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=0.9,
                )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, cell[dims[0]])
            ax.set_ylim(0, cell[dims[1]])
            ax.grid(alpha=0.15)
            ax.set_title(title if row == 0 else "")
            ax.set_xlabel(("x (A)" if dims[0] == 0 else "y (A)"))
            ax.set_ylabel(("y (A)" if dims[1] == 1 else "z (A)"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    fig.suptitle("Trajectory snapshots: xy projection (top), xz projection (bottom)", y=0.985)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=2.4, w_pad=2.0)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = parse_log(run_dir)
    frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
    if len(frames) != len(data):
        raise RuntimeError(f"Frame/log length mismatch: {len(frames)} frames vs {len(data)} log rows")
    mins = min_distances(frames)
    max_forces = max_force_norms(frames)
    natoms = len(frames[0])

    plot_temperature_energy(data, out_dir / "temperature_energy_100ps.png")
    plot_potential_kinetic_force(data, max_forces, out_dir / "epot_ekin_maxforce_100ps.png")
    plot_min_distance(data[:, 0], mins, out_dir / "min_pair_distance_100ps.png")
    plot_snapshots(frames, data, out_dir / "structure_snapshots_0_50_100ps.png")

    rows = [
        stat_row("T_K", data[:, 4]),
        stat_row("Epot_atom", data[:, 2] / natoms),
        stat_row("Ekin", data[:, 3]),
        stat_row("Fmax", max_forces),
        stat_row("dmin", mins),
    ]
    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"plots_dir: {out_dir}")


if __name__ == "__main__":
    main()
