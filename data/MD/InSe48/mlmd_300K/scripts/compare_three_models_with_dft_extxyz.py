#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread, read


def parse_ml_log(run_dir: Path) -> np.ndarray:
    segments: list[np.ndarray] = []
    current = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        if line.startswith("Time"):
            if current:
                segments.append(np.array(current, dtype=float))
            current = []
        elif line.strip():
            current.append([float(x) for x in line.split()])
    if current:
        segments.append(np.array(current, dtype=float))
    if not segments:
        raise RuntimeError(f"No log rows in {run_dir / 'mlmd.log'}")
    return segments[-1]


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


def dmin_series(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        d = atoms.get_all_distances(mic=True)
        vals.append(float(d[np.triu_indices(len(atoms), 1)].min()))
    return np.array(vals)


def fmax_series(frames) -> np.ndarray:
    vals = []
    for atoms in frames:
        f = atoms.get_forces()
        vals.append(float(np.linalg.norm(f, axis=1).max()))
    return np.array(vals)


def partial_rdf(frames, pair: tuple[str, str], rmax: float, dr: float) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, rmax + dr, dr)
    hist = np.zeros(len(bins) - 1)
    a, b = pair
    for atoms in frames:
        symbols = np.array(atoms.get_chemical_symbols())
        idx_a = np.where(symbols == a)[0]
        idx_b = np.where(symbols == b)[0]
        dist = atoms.get_all_distances(mic=True)
        if a == b:
            vals = dist[np.ix_(idx_a, idx_a)][np.triu_indices(len(idx_a), 1)]
        else:
            vals = dist[np.ix_(idx_a, idx_b)].ravel()
        hist += np.histogram(vals, bins=bins)[0]

    centers = 0.5 * (bins[:-1] + bins[1:])
    shell = 4.0 * np.pi * centers**2 * dr
    volume = float(frames[0].get_volume())
    symbols = np.array(frames[0].get_chemical_symbols())
    n_a = int(np.sum(symbols == a))
    n_b = int(np.sum(symbols == b))
    density_b = (n_b if a != b else max(n_b - 1, 1)) / volume
    norm = len(frames) * n_a * density_b * shell
    if a == b:
        norm *= 0.5
    rdf = np.divide(hist, norm, out=np.zeros_like(hist), where=norm > 0)
    return centers, rdf


def plot_time_series(series: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    specs = [
        ("T_K", "Temperature (K)", "temperature_compare.png"),
        ("Epot_atom", "Epot/atom (eV)", "epot_atom_compare.png"),
        ("Ekin", "Ekin (eV)", "ekin_compare.png"),
        ("dmin", "Minimum pair distance (A)", "dmin_compare.png"),
        ("Fmax", "Fmax (eV/A)", "fmax_compare.png"),
    ]
    for key, ylabel, name in specs:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for label, data in series.items():
            if key in data:
                ax.plot(data["time"], data[key], lw=1.2, label=label)
        ax.set_xlabel("Time since selected start (ps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)


def plot_rdf(
    rdf_frames: dict[str, list],
    pairs: list[tuple[str, str]],
    out_dir: Path,
    rmax: float,
    dr: float,
) -> None:
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, pair in zip(axes.ravel(), pairs):
        for label, frames in rdf_frames.items():
            r, g = partial_rdf(frames, pair, rmax, dr)
            ax.plot(r, g, lw=1.2, label=label)
        ax.set_title("-".join(pair))
        ax.set_xlim(1.0, rmax)
        ax.set_xlabel("r (A)")
        ax.set_ylabel("g(r)")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(pairs) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "partial_rdf_compare.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-extxyz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dft-start-index", type=int, default=1000)
    parser.add_argument("--max-rdf-frames", type=int, default=101)
    parser.add_argument("--rmax", type=float, default=6.0)
    parser.add_argument("--dr", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "finetuned_balanced": Path("runs/run_001_frame1001_300K_10ps"),
        "official_r2scan": Path("runs/run_002_official_r2scan_frame1001_300K_10ps"),
        "baseline_random_e20": Path("runs/run_003_baseline_random_e20_frame1001_300K_10ps"),
    }

    series: dict[str, dict[str, np.ndarray]] = {}
    rdf_frames: dict[str, list] = {}
    rows = []

    for label, run_dir in run_dirs.items():
        log = parse_ml_log(run_dir)
        frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
        natoms = len(frames[0])
        data = {
            "time": log[:, 0],
            "T_K": log[:, 4],
            "Epot_atom": log[:, 2] / natoms,
            "Ekin": log[:, 3],
            "dmin": dmin_series(frames),
            "Fmax": fmax_series(frames),
        }
        series[label] = data
        rdf_frames[label] = frames[: args.max_rdf_frames]
        for key in ("T_K", "Epot_atom", "Ekin", "dmin", "Fmax"):
            rows.append(stat_row(label, key, data[key]))

    dft_frames_all = list(iread(str(args.dft_extxyz), index=slice(args.dft_start_index, None, None)))
    dft_frames_rdf = dft_frames_all[: args.max_rdf_frames * 10 : 10]
    natoms = len(dft_frames_all[0])
    dft_time = np.arange(len(dft_frames_all), dtype=float) * 0.001
    dft_epot = np.array([atoms.get_potential_energy() / natoms for atoms in dft_frames_all])
    dft_fmax = fmax_series(dft_frames_all)
    dft_dmin = dmin_series(dft_frames_all)
    dft_temp = np.array([float(atoms.info.get("temperature_K", np.nan)) for atoms in dft_frames_all])
    dft_ekin = np.full(len(dft_frames_all), np.nan)
    data = {
        "time": dft_time,
        "T_K": dft_temp,
        "Epot_atom": dft_epot,
        "Ekin": dft_ekin,
        "dmin": dft_dmin,
        "Fmax": dft_fmax,
    }
    series["DFT_r2SCAN_from_frame1001"] = data
    rdf_frames["DFT_r2SCAN_from_frame1001"] = dft_frames_rdf
    for key in ("T_K", "Epot_atom", "dmin", "Fmax"):
        rows.append(stat_row("DFT_r2SCAN_from_frame1001", key, data[key]))

    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elements = sorted(set(dft_frames_all[0].get_chemical_symbols()))
    pairs = list(combinations_with_replacement(elements, 2))
    plot_time_series(series, out_dir)
    plot_rdf(rdf_frames, pairs, out_dir, args.rmax, args.dr)

    print(f"wrote: {out_dir}")
    print(f"stats: {out_dir / 'summary_stats.csv'}")


if __name__ == "__main__":
    main()
