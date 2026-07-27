#!/usr/bin/env python
from __future__ import annotations

import csv
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread, read


DFT_EXTXYZ = Path(
    "/inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/"
    "ml_test_dataset_r2scan_HgInSn48/r2scan_300K_5ps/"
    "HgInSn48_r2scan_300K_5ps_prod_after500fs_all.extxyz"
)
DFT_START_INDEX = 546


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
    return segments[-1]


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


def stat_row(label: str, quantity: str, values: np.ndarray) -> dict[str, str]:
    values = values[np.isfinite(values)]
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


def partial_rdf(frames, pair: tuple[str, str], rmax: float = 6.0, dr: float = 0.05):
    bins = np.arange(0.0, rmax + dr, dr)
    hist = np.zeros(len(bins) - 1)
    a, b = pair
    for atoms in frames:
        symbols = np.array(atoms.get_chemical_symbols())
        ia = np.where(symbols == a)[0]
        ib = np.where(symbols == b)[0]
        dist = atoms.get_all_distances(mic=True)
        if a == b:
            vals = dist[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)]
        else:
            vals = dist[np.ix_(ia, ib)].ravel()
        hist += np.histogram(vals, bins=bins)[0]
    centers = 0.5 * (bins[:-1] + bins[1:])
    shell = 4.0 * np.pi * centers**2 * dr
    symbols = np.array(frames[0].get_chemical_symbols())
    n_a = int(np.sum(symbols == a))
    n_b = int(np.sum(symbols == b))
    density_b = (n_b if a != b else max(n_b - 1, 1)) / float(frames[0].get_volume())
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
            if key in data and np.isfinite(data[key]).any():
                ax.plot(data["time"], data[key], lw=1.2, label=label)
        ax.set_xlabel("Time since source_step=1047 start (ps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)


def plot_rdf(rdf_frames: dict[str, list], out_dir: Path) -> None:
    pairs = [("Hg", "Hg"), ("Hg", "In"), ("Hg", "Sn"), ("In", "In"), ("In", "Sn"), ("Sn", "Sn")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for ax, pair in zip(axes.ravel(), pairs):
        for label, frames in rdf_frames.items():
            r, g = partial_rdf(frames, pair)
            ax.plot(r, g, lw=1.2, label=label)
        ax.set_title("-".join(pair))
        ax.set_xlim(1.5, 6.0)
        ax.set_xlabel("r (A)")
        ax.set_ylabel("g(r)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "partial_rdf_compare.png", dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("runs/comparison_step1047_three_models_vs_dft")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "base_finetuned_balanced": Path("runs/run_005_step1047_base_300K_10ps"),
        "official_r2scan": Path("runs/run_006_step1047_official_r2scan_300K_10ps"),
        "baseline_random_e20": Path("runs/run_007_step1047_baseline_random_e20_300K_10ps"),
    }
    series: dict[str, dict[str, np.ndarray]] = {}
    rdf_frames: dict[str, list] = {}
    rows = []
    for label, run_dir in runs.items():
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
        rdf_frames[label] = frames[:101]
        for key in ("T_K", "Epot_atom", "Ekin", "dmin", "Fmax"):
            rows.append(stat_row(label, key, data[key]))

    dft_frames = list(iread(str(DFT_EXTXYZ), index=slice(DFT_START_INDEX, None, None)))
    natoms = len(dft_frames[0])
    dft_data = {
        "time": np.arange(len(dft_frames), dtype=float) * 0.001,
        "T_K": np.array([float(a.info.get("temperature_K", np.nan)) for a in dft_frames]),
        "Epot_atom": np.array([a.get_potential_energy() / natoms for a in dft_frames]),
        "Ekin": np.full(len(dft_frames), np.nan),
        "dmin": dmin_series(dft_frames),
        "Fmax": fmax_series(dft_frames),
    }
    series["DFT_r2SCAN_from_source_step1047"] = dft_data
    rdf_frames["DFT_r2SCAN_from_source_step1047"] = dft_frames[:1010:10]
    for key in ("T_K", "Epot_atom", "dmin", "Fmax"):
        rows.append(stat_row("DFT_r2SCAN_from_source_step1047", key, dft_data[key]))

    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_time_series(series, out_dir)
    plot_rdf(rdf_frames, out_dir)
    print(f"wrote: {out_dir}")
    print(f"stats: {out_dir / 'summary_stats.csv'}")


if __name__ == "__main__":
    main()
