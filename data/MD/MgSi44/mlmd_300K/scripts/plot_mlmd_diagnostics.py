#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


COLORS = {
    "Mg": "#2f80ed",
    "Si": "#d95f02",
}


def parse_log_segments(log_path: Path) -> list[np.ndarray]:
    lines = log_path.read_text().splitlines()
    header_idx = [i for i, line in enumerate(lines) if line.startswith("Time[ps]")]
    segments: list[np.ndarray] = []
    for idx, start in enumerate(header_idx):
        end = header_idx[idx + 1] if idx + 1 < len(header_idx) else len(lines)
        rows = []
        for line in lines[start + 1 : end]:
            if line.strip():
                rows.append([float(x) for x in line.split()])
        if rows:
            segments.append(np.array(rows, dtype=float))
    return segments


def min_distances(frames) -> np.ndarray:
    mins = []
    for atoms in frames:
        distances = atoms.get_all_distances(mic=True)
        iu = np.triu_indices(len(atoms), 1)
        mins.append(float(distances[iu].min()))
    return np.array(mins, dtype=float)


def max_force_norms(frames) -> np.ndarray:
    max_forces = []
    for atoms in frames:
        forces = atoms.get_forces()
        norms = np.linalg.norm(forces, axis=1)
        max_forces.append(float(norms.max()))
    return np.array(max_forces, dtype=float)


def suffix_from_time(data: np.ndarray) -> str:
    end_ps = float(data[-1, 0])
    if end_ps >= 99.0:
        return "100ps"
    if end_ps >= 9.0:
        return "10ps"
    return f"{end_ps:.1f}ps".replace(".", "p")


def plot_temperature_energy(data: np.ndarray, out_path: Path) -> None:
    t = data[:, 0]
    etot = data[:, 1]
    epot = data[:, 2]
    temp = data[:, 4]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(t, temp, color="#1f77b4", lw=1.2)
    axes[0].axhline(300, color="#555555", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("MgSi44 CHGNet ML-MD 300 K NVT diagnostics")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, epot, label="Epot", color="#d62728", lw=1.2)
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
    axes[0].plot(t, epot, color="#d62728", lw=1.2)
    axes[0].set_ylabel("Epot (eV)")
    axes[0].set_title("Potential energy, kinetic energy, and maximum force")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, ekin, color="#ff7f0e", lw=1.2)
    axes[1].set_ylabel("Ekin (eV)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, max_forces, color="#6a3d9a", lw=1.2)
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel("max |F| (eV/A)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_min_distance(times: np.ndarray, mins: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(times, mins, color="#6a3d9a", lw=1.2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Minimum pair distance (A)")
    ax.set_title("Closest atom pair over trajectory")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_snapshots(frames, data: np.ndarray, out_path: Path) -> None:
    end_ps = float(data[-1, 0])
    sample_ps = [0.0, 0.5 * end_ps, end_ps]
    times = data[:, 0]
    indices = [int(np.argmin(np.abs(times - t))) for t in sample_ps]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    elements = sorted(set(frames[0].get_chemical_symbols()))
    for col, frame_idx in enumerate(indices):
        atoms = frames[frame_idx]
        symbols = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = atoms.cell.lengths()
        title = f"{times[frame_idx]:.1f} ps"

        for row, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[row, col]
            for element in elements:
                mask = symbols == element
                ax.scatter(
                    pos[mask, dims[0]],
                    pos[mask, dims[1]],
                    s=44,
                    c=COLORS.get(element, "#777777"),
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
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), frameon=False)
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

    log_segments = parse_log_segments(run_dir / "mlmd.log")
    if not log_segments:
        raise RuntimeError(f"No MD log segments found in {run_dir / 'mlmd.log'}")

    data = log_segments[-1]
    frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
    if len(frames) != len(data):
        raise RuntimeError(f"Frame/log length mismatch: {len(frames)} frames vs {len(data)} log rows")

    mins = min_distances(frames)
    max_forces = max_force_norms(frames)
    suffix = suffix_from_time(data)
    plot_temperature_energy(data, out_dir / f"temperature_energy_{suffix}.png")
    plot_potential_kinetic_force(data, max_forces, out_dir / f"epot_ekin_maxforce_{suffix}.png")
    plot_min_distance(data[:, 0], mins, out_dir / f"min_pair_distance_{suffix}.png")
    plot_snapshots(frames, data, out_dir / f"structure_snapshots_0_{suffix}.png")

    summary = {
        "log_rows": int(len(data)),
        "frames": int(len(frames)),
        "time_ps_start": float(data[0, 0]),
        "time_ps_end": float(data[-1, 0]),
        "temperature_K_min": float(data[:, 4].min()),
        "temperature_K_max": float(data[:, 4].max()),
        "temperature_K_mean": float(data[:, 4].mean()),
        "epot_eV_start": float(data[0, 2]),
        "epot_eV_end": float(data[-1, 2]),
        "epot_eV_per_atom_mean": float((data[:, 2] / len(frames[0])).mean()),
        "epot_eV_per_atom_std": float((data[:, 2] / len(frames[0])).std()),
        "min_pair_distance_A_min": float(mins.min()),
        "min_pair_distance_A_mean": float(mins.mean()),
        "min_pair_distance_A_final": float(mins[-1]),
        "max_force_eV_per_A_mean": float(max_forces.mean()),
        "max_force_eV_per_A_max": float(max_forces.max()),
        "max_force_eV_per_A_final": float(max_forces[-1]),
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"plots_dir: {out_dir}")


if __name__ == "__main__":
    main()
