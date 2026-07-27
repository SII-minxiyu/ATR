#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.io.vasp import read_vasp_xdatcar


MD_RE = re.compile(
    r"^\s*(?P<step>\d+)\s+T=\s*(?P<T>[-+0-9.]+)\.\s+E=\s*(?P<E>[-+.0-9E]+)"
    r"\s+F=\s*(?P<F>[-+.0-9E]+).*?EK=\s*(?P<EK>[-+.0-9E]+)"
)


def parse_ml_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        if line.strip() and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=float)


def parse_dft_oszicar(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = MD_RE.match(line)
        if match:
            rows.append(
                [
                    float(match.group("step")),
                    float(match.group("T")),
                    float(match.group("E")),
                    float(match.group("F")),
                    float(match.group("EK")),
                ]
            )
    return np.array(rows, dtype=float)


def parse_outcar_fmax(path: Path, natoms: int) -> np.ndarray:
    fmax = []
    with path.open(errors="ignore") as handle:
        in_block = False
        skip = 0
        forces = []
        for line in handle:
            if "TOTAL-FORCE (eV/Angst)" in line:
                in_block = True
                skip = 1
                forces = []
                continue
            if not in_block:
                continue
            if skip:
                skip -= 1
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    pass
            if len(forces) == natoms:
                norms = np.linalg.norm(np.array(forces), axis=1)
                fmax.append(float(norms.max()))
                in_block = False
    return np.array(fmax, dtype=float)


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
    out = []
    for atoms in frames:
        dist = atoms.get_all_distances(mic=True)
        out.append(float(dist[np.triu_indices(len(atoms), 1)].min()))
    return np.array(out, dtype=float)


def fmax_series(frames) -> np.ndarray:
    out = []
    for atoms in frames:
        forces = atoms.get_forces()
        out.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(out, dtype=float)


def partial_rdf(frames, pair: tuple[str, str], rmax: float, dr: float) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, rmax + dr, dr)
    hist = np.zeros(len(bins) - 1, dtype=float)
    a, b = pair
    for atoms in frames:
        symbols = np.array(atoms.get_chemical_symbols())
        idx_a = np.where(symbols == a)[0]
        idx_b = np.where(symbols == b)[0]
        dist = atoms.get_all_distances(mic=True)
        if a == b:
            vals = dist[np.ix_(idx_a, idx_a)]
            iu = np.triu_indices(len(idx_a), 1)
            vals = vals[iu]
        else:
            vals = dist[np.ix_(idx_a, idx_b)].ravel()
        hist += np.histogram(vals, bins=bins)[0]

    centers = 0.5 * (bins[:-1] + bins[1:])
    shell = 4.0 * np.pi * centers**2 * dr
    volume = float(frames[0].get_volume())
    symbols = np.array(frames[0].get_chemical_symbols())
    n_a = int(np.sum(symbols == a))
    n_b = int(np.sum(symbols == b))
    n_ref = n_a if a != b else n_a
    density_b = (n_b if a != b else max(n_b - 1, 1)) / volume
    norm = len(frames) * n_ref * density_b * shell
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
            if key not in data:
                continue
            ax.plot(data["time"], data[key], lw=1.2, label=label)
        ax.set_xlabel("Time since selected start (ps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)


def plot_rdf(rdfs: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]], out_dir: Path) -> None:
    pairs = [("Hg", "Hg"), ("Hg", "In"), ("Hg", "Sn"), ("In", "In"), ("In", "Sn"), ("Sn", "Sn")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for ax, pair in zip(axes.ravel(), pairs):
        for label, rdf_by_pair in rdfs.items():
            r, g = rdf_by_pair[pair]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/comparison_step2501_three_models_vs_dft"))
    parser.add_argument("--dft-run004", type=Path, required=True)
    parser.add_argument("--dft-start-index", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=251)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_runs = {
        "finetuned_balanced": Path("runs/run_002_step2501_300K_10ps"),
        "official_r2scan": Path("runs/run_003_official_chgnet_r2scan_step2501_300K_10ps"),
        "baseline_random_e20": Path("runs/run_004_baseline_random_e20_step2501_300K_10ps"),
    }

    series: dict[str, dict[str, np.ndarray]] = {}
    stats = []
    rdf_frames: dict[str, list] = {}

    for label, run_dir in ml_runs.items():
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
        rdf_frames[label] = frames[: args.max_frames]
        for key in ("T_K", "Epot_atom", "Ekin", "dmin", "Fmax"):
            stats.append(stat_row(label, key, data[key]))

    dft_log = parse_dft_oszicar(args.dft_run004 / "OSZICAR")
    dft_slice = dft_log[args.dft_start_index :]
    dft_frames_all = read_vasp_xdatcar(args.dft_run004 / "XDATCAR", index=slice(args.dft_start_index, None, None))
    if not isinstance(dft_frames_all, list):
        dft_frames_all = [dft_frames_all]
    natoms = len(dft_frames_all[0])
    dft_fmax = parse_outcar_fmax(args.dft_run004 / "OUTCAR", natoms)[args.dft_start_index :]
    dft_time = np.arange(len(dft_slice), dtype=float) * 0.001
    data = {
        "time": dft_time,
        "T_K": dft_slice[:, 1],
        "Epot_atom": dft_slice[:, 3] / natoms,
        "Ekin": dft_slice[:, 4],
        "dmin": dmin_series(dft_frames_all),
        "Fmax": dft_fmax,
    }
    min_len = min(len(v) for v in data.values())
    data = {k: v[:min_len] for k, v in data.items()}
    series["DFT_r2SCAN_2.5-5ps"] = data
    # ML trajectories are written every 10 fs while XDATCAR stores every 1 fs.
    # Downsample DFT by 10 so the RDF uses the same approximate time span.
    rdf_frames["DFT_r2SCAN_2.5-5ps"] = dft_frames_all[: args.max_frames * 10 : 10]
    for key in ("T_K", "Epot_atom", "Ekin", "dmin", "Fmax"):
        stats.append(stat_row("DFT_r2SCAN_2.5-5ps", key, data[key]))

    with (out_dir / "summary_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    plot_time_series(series, out_dir)

    rdfs = {}
    pairs = [("Hg", "Hg"), ("Hg", "In"), ("Hg", "Sn"), ("In", "In"), ("In", "Sn"), ("Sn", "Sn")]
    for label, frames in rdf_frames.items():
        rdfs[label] = {pair: partial_rdf(frames, pair, 6.0, 0.05) for pair in pairs}
    plot_rdf(rdfs, out_dir)

    print(f"wrote: {out_dir}")
    print(f"stats: {out_dir / 'summary_stats.csv'}")


if __name__ == "__main__":
    main()
