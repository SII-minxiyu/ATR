#!/usr/bin/env python
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


NATOMS = 48
DFT_ROOT = Path(
    "/inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md"
)
DFT_SEGMENTS = [
    # source_step=2501 is 2000 steps from run_002 plus frame 501 from run_004.
    (DFT_ROOT / "run_004_r2scan_nvt_300K_extend_3ps", 500, None),
    (DFT_ROOT / "run_006_r2scan_nvt_300K_extend_7p5ps", 0, None),
]

RUNS = {
    "finetuned_balanced": Path("runs/run_008_step2501_base_300K_100ps"),
    "official_r2scan": Path("runs/run_009_step2501_official_r2scan_300K_100ps"),
    "baseline_random_e20": Path("runs/run_010_step2501_baseline_random_e20_300K_100ps"),
}

PAIR_OUTPUTS = [
    (
        "official_r2scan",
        Path("runs/comparison_step2501_base_vs_official_100ps_vs_dft"),
        {
            "T_K": "temperature_compare_with_dft12p5.png",
            "Epot_atom": "epot_atom_compare_with_dft12p5.png",
            "Ekin": "ekin_compare_with_dft12p5.png",
            "Fmax": "fmax_compare_with_dft12p5.png",
            "dmin": "dmin_compare_with_dft12p5.png",
        },
    ),
    (
        "baseline_random_e20",
        Path("runs/comparison_step2501_base_vs_baseline_100ps"),
        {
            "T_K": "temperature_compare_with_dft12p5.png",
            "Epot_atom": "epot_atom_compare_with_dft12p5.png",
            "Ekin": "ekin_compare_with_dft12p5.png",
            "Fmax": "fmax_compare_with_dft12p5.png",
            "dmin": "dmin_compare_with_dft12p5.png",
        },
    ),
]

COLORS = {
    "DFT_r2SCAN_12p5ps_from_step2501": "#111111",
    "finetuned_balanced": "#1f77b4",
    "official_r2scan": "#2ca02c",
    "baseline_random_e20": "#d62728",
}
ZORDER = {
    "DFT_r2SCAN_12p5ps_from_step2501": 3,
    "finetuned_balanced": 2,
    "official_r2scan": 1,
    "baseline_random_e20": 1,
}


def parse_mlmd_log(run_dir: Path) -> np.ndarray:
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


def load_fmax_csv(path: Path) -> np.ndarray:
    values = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row["fmax_eV_per_A"]))
    if not values:
        raise RuntimeError(f"No Fmax rows found in {path}")
    return np.array(values, dtype=float)


def parse_oszicar(path: Path) -> dict[str, np.ndarray]:
    pattern = re.compile(
        r"^\s*\d+\s+T=\s*([+-]?\d+(?:\.\d*)?)\.\s+E=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
        r"\s+F=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
        r".*?EK=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
    )
    temp, etot, epot, ekin = [], [], [], []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            t, e, f, ek = match.groups()
            temp.append(float(t))
            etot.append(float(e))
            epot.append(float(f))
            ekin.append(float(ek))
    if not temp:
        raise RuntimeError(f"No MD rows parsed from {path}")
    return {
        "T_K": np.array(temp, dtype=float),
        "Etot": np.array(etot, dtype=float),
        "Epot": np.array(epot, dtype=float),
        "Ekin": np.array(ekin, dtype=float),
    }


def parse_outcar_fmax(path: Path, natoms: int = NATOMS) -> np.ndarray:
    vals = []
    with path.open(errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            if "TOTAL-FORCE (eV/Angst)" not in line:
                continue
            next(lines, None)
            norms = []
            for _ in range(natoms):
                parts = next(lines).split()
                if len(parts) >= 6:
                    fx, fy, fz = map(float, parts[-3:])
                    norms.append((fx * fx + fy * fy + fz * fz) ** 0.5)
            if norms:
                vals.append(max(norms))
    if not vals:
        raise RuntimeError(f"No force blocks parsed from {path}")
    return np.array(vals, dtype=float)


def load_dft() -> dict[str, np.ndarray]:
    pieces = {"T_K": [], "Epot": [], "Ekin": [], "Fmax": [], "dmin": []}
    for seg_dir, start, stop in DFT_SEGMENTS:
        osz = parse_oszicar(seg_dir / "OSZICAR")
        frames = read(seg_dir / "XDATCAR", ":")
        fmax = parse_outcar_fmax(seg_dir / "OUTCAR")
        n = min(len(osz["T_K"]), len(frames), len(fmax))
        sl = slice(start, stop if stop is not None else n)
        pieces["T_K"].append(osz["T_K"][:n][sl])
        pieces["Epot"].append(osz["Epot"][:n][sl])
        pieces["Ekin"].append(osz["Ekin"][:n][sl])
        pieces["Fmax"].append(fmax[:n][sl])
        pieces["dmin"].append(dmin_series(frames[:n][sl]))

    data = {key: np.concatenate(values) for key, values in pieces.items()}
    n = len(data["T_K"])
    data["time"] = np.arange(n, dtype=float) * 0.001
    data["Epot_atom"] = data["Epot"] / NATOMS
    return data


def load_mlmd(label: str, run_dir: Path) -> dict[str, np.ndarray]:
    log = parse_mlmd_log(run_dir)
    frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
    if len(log) != len(frames):
        raise RuntimeError(f"{label}: {len(log)} log rows vs {len(frames)} frames")
    fmax_csv = run_dir / "fmax_timeseries.csv"
    fmax = load_fmax_csv(fmax_csv) if fmax_csv.exists() else fmax_series(frames)
    if len(fmax) != len(log):
        raise RuntimeError(f"{label}: {len(fmax)} Fmax rows vs {len(log)} log rows")
    return {
        "time": log[:, 0],
        "T_K": log[:, 4],
        "Epot_atom": log[:, 2] / len(frames[0]),
        "Ekin": log[:, 3],
        "Fmax": fmax,
        "dmin": dmin_series(frames),
    }


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


def plot_series(
    series: dict[str, dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float | None, float | None] | None = None,
) -> None:
    # Draw bottom layer first, DFT last, while keeping legend order as DFT/base/other.
    draw_order = list(reversed(list(series.keys())))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label in draw_order:
        data = series[label]
        ax.plot(
            data["time"],
            data[key],
            lw=1.15 if label == "DFT_r2SCAN_12p5ps_from_step2501" else 1.0,
            label=label,
            color=COLORS[label],
            zorder=ZORDER[label],
        )
    if key == "T_K":
        ax.axhline(300, color="#666666", lw=1.0, ls="--", alpha=0.7, zorder=0)
    ax.set_xlim(-4, 104)
    ax.set_xticks(np.arange(0, 101, 20))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time since source_step=2501 start (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(label) for label in series.keys()]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_pair(other_label: str, out_dir: Path, filenames: dict[str, str], all_series: dict[str, dict[str, np.ndarray]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = {
        "DFT_r2SCAN_12p5ps_from_step2501": all_series["DFT_r2SCAN_12p5ps_from_step2501"],
        "finetuned_balanced": all_series["finetuned_balanced"],
        other_label: all_series[other_label],
    }
    specs = {
        "T_K": "Temperature (K)",
        "Epot_atom": "Epot/atom (eV)",
        "Ekin": "Ekin (eV)",
        "Fmax": "max |F| (eV/A)",
        "dmin": "Minimum pair distance (A)",
    }
    for key, ylabel in specs.items():
        ylim = None
        if out_dir.name == "comparison_step2501_base_vs_baseline_100ps" and key == "dmin":
            ylim = (None, 5)
        if out_dir.name == "comparison_step2501_base_vs_baseline_100ps" and key == "Fmax":
            ylim = (None, 15)
        plot_series(ordered, key, ylabel, out_dir / filenames[key], ylim=ylim)

    stats = []
    for label, data in ordered.items():
        for key in specs:
            stats.append(stat_row(label, key, data[key]))
    with (out_dir / "summary_stats_with_dft12p5.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)


def main() -> None:
    all_series = {
        "DFT_r2SCAN_12p5ps_from_step2501": load_dft(),
        "finetuned_balanced": load_mlmd("finetuned_balanced", RUNS["finetuned_balanced"]),
        "official_r2scan": load_mlmd("official_r2scan", RUNS["official_r2scan"]),
        "baseline_random_e20": load_mlmd("baseline_random_e20", RUNS["baseline_random_e20"]),
    }
    for other_label, out_dir, filenames in PAIR_OUTPUTS:
        write_pair(other_label, out_dir, filenames, all_series)
        print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
