#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
OUT_DIR = ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/picture"

COLORS = {
    "Teacher": "#FF6445",
    "Baseline": "#4497F2",
}

SYSTEMS = {
    "HgInSn": {
        "target_T": 300,
        "runs": {
            "Teacher": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_009_step2501_official_r2scan_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_010_step2501_baseline_random_e20_300K_100ps",
        },
    },
    "InSe": {
        "target_T": 300,
        "runs": {
            "Teacher": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_005_official_r2scan_frame1001_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
    },
    "MgSi": {
        "target_T": 300,
        "runs": {
            "Teacher": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_005_frame1001_official_r2scan_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
    },
}


def parse_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"No rows found in {run_dir / 'mlmd.log'}")
    return np.array(rows, dtype=float)


def fmax_series(run_dir: Path) -> np.ndarray:
    values = []
    for atoms in read(run_dir / "mlmd_trajectory.extxyz", ":"):
        forces = atoms.get_forces()
        values.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(values, dtype=float)


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    log = parse_log(run_dir)
    natoms = len(read(run_dir / "final_structure.extxyz"))
    return {
        "time": log[:, 0],
        "temperature": log[:, 4],
        "epot_atom": log[:, 2] / natoms,
        "ekin": log[:, 3],
        "fmax": fmax_series(run_dir),
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


def plot_metric(system_name: str, target_T: float, series: dict[str, dict[str, np.ndarray]], metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label in ("Baseline", "Teacher"):
        data = series[label]
        ax.plot(data["time"], data[metric], color=COLORS[label], lw=1.1)

    if metric == "temperature":
        ax.axhline(target_T, color="#666666", lw=1.0, ls="--", alpha=0.75, zorder=0)

    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()

    png = OUT_DIR / f"{system_name}_300K_Teacher_vs_Baseline_{metric}.png"
    pdf = OUT_DIR / f"{system_name}_300K_Teacher_vs_Baseline_{metric}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "ekin": "Ekin (eV)",
        "epot_atom": "Epot/atom (eV)",
        "fmax": "max |F| (eV/A)",
        "temperature": "Temperature (K)",
    }
    for system_name, system in SYSTEMS.items():
        print(f"loading {system_name}")
        series = {label: load_run(run_dir) for label, run_dir in system["runs"].items()}
        for metric, ylabel in specs.items():
            plot_metric(system_name, system["target_T"], series, metric, ylabel)
    print(f"wrote Teacher-vs-Baseline figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
