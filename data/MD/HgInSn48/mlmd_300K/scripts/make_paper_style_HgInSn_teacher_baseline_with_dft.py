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


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
WORK_DIR = ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes"
OUT_DIR = WORK_DIR / "picture"
DFT_ROOT = ROOT / "luyouqi/md_HgInSn_48atom_vasp/reference_dft_md"

NATOMS = 48
DFT_SEGMENTS = [
    (DFT_ROOT / "run_004_r2scan_nvt_300K_extend_3ps", 500, None),
    (DFT_ROOT / "run_006_r2scan_nvt_300K_extend_7p5ps", 0, None),
]

RUNS = {
    "Teacher": WORK_DIR / "runs/run_009_step2501_official_r2scan_300K_100ps",
    "Baseline": WORK_DIR / "runs/run_010_step2501_baseline_random_e20_300K_100ps",
}

COLORS = {
    "DFT-AIMD": "#111111",
    "Teacher": "#FF6445",
    "Baseline": "#4497F2",
}


def parse_oszicar(path: Path) -> dict[str, np.ndarray]:
    pattern = re.compile(
        r"^\s*\d+\s+T=\s*([+-]?\d+(?:\.\d*)?)\.\s+E=\s*([+-]?(?:\.\d+|\d+\.\d+|\d+)(?:E[+-]\d+)?)"
        r"\s+F=\s*([+-]?(?:\.\d+|\d+\.\d+|\d+)(?:E[+-]\d+)?)"
        r".*?EK=\s*([+-]?(?:\.\d+|\d+\.\d+|\d+)(?:E[+-]\d+)?)"
    )
    temp, epot, ekin = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            t, _, f, ek = match.groups()
            temp.append(float(t))
            epot.append(float(f))
            ekin.append(float(ek))
    if not temp:
        raise RuntimeError(f"No MD rows parsed from {path}")
    return {
        "temperature": np.array(temp, dtype=float),
        "epot_atom": np.array(epot, dtype=float) / NATOMS,
        "ekin": np.array(ekin, dtype=float),
    }


def parse_outcar_fmax(path: Path, natoms: int = NATOMS) -> np.ndarray:
    values = []
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
                values.append(max(norms))
    if not values:
        raise RuntimeError(f"No force blocks parsed from {path}")
    return np.array(values, dtype=float)


def load_dft() -> dict[str, np.ndarray]:
    pieces = {"temperature": [], "epot_atom": [], "ekin": [], "fmax": []}
    for seg_dir, start, stop in DFT_SEGMENTS:
        osz = parse_oszicar(seg_dir / "OSZICAR")
        fmax = parse_outcar_fmax(seg_dir / "OUTCAR")
        n = min(len(osz["temperature"]), len(fmax))
        sl = slice(start, stop if stop is not None else n)
        for key in ("temperature", "epot_atom", "ekin"):
            pieces[key].append(osz[key][:n][sl])
        pieces["fmax"].append(fmax[:n][sl])
    data = {key: np.concatenate(values) for key, values in pieces.items()}
    data["time"] = np.arange(len(data["temperature"]), dtype=float) * 0.001
    return data


def parse_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"No rows found in {run_dir / 'mlmd.log'}")
    return np.array(rows, dtype=float)


def load_fmax_csv(path: Path) -> np.ndarray:
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["fmax_eV_per_A"]))
    if not values:
        raise RuntimeError(f"No rows found in {path}")
    return np.array(values, dtype=float)


def fmax_from_extxyz(run_dir: Path) -> np.ndarray:
    values = []
    for atoms in read(run_dir / "mlmd_trajectory.extxyz", ":"):
        forces = atoms.get_forces()
        values.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(values, dtype=float)


def load_mlmd(run_dir: Path) -> dict[str, np.ndarray]:
    log = parse_log(run_dir)
    natoms = len(read(run_dir / "final_structure.extxyz"))
    fmax_csv = run_dir / "fmax_timeseries.csv"
    fmax = load_fmax_csv(fmax_csv) if fmax_csv.exists() else fmax_from_extxyz(run_dir)
    return {
        "time": log[:, 0],
        "temperature": log[:, 4],
        "epot_atom": log[:, 2] / natoms,
        "ekin": log[:, 3],
        "fmax": fmax,
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(series: dict[str, dict[str, np.ndarray]], metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, lw in (("Baseline", 1.05), ("Teacher", 1.1), ("DFT-AIMD", 1.45)):
        data = series[label]
        ax.plot(data["time"], data[metric], color=COLORS[label], lw=lw)

    if metric == "temperature":
        ax.axhline(300, color="#666666", lw=1.0, ls="--", alpha=0.75, zorder=0)

    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()

    png = OUT_DIR / f"HgInSn_300K_Teacher_vs_Baseline_{metric}.png"
    pdf = OUT_DIR / f"HgInSn_300K_Teacher_vs_Baseline_{metric}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = {
        "DFT-AIMD": load_dft(),
        "Teacher": load_mlmd(RUNS["Teacher"]),
        "Baseline": load_mlmd(RUNS["Baseline"]),
    }
    specs = {
        "ekin": "Ekin (eV)",
        "epot_atom": "Epot/atom (eV)",
        "fmax": "max |F| (eV/A)",
        "temperature": "Temperature (K)",
    }
    for metric, ylabel in specs.items():
        plot_metric(series, metric, ylabel)
    print(f"wrote HgInSn DFT + Teacher + Baseline figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
