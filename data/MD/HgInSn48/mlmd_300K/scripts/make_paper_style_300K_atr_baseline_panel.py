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
    "ATR": "#FF8E00",
    "Baseline": "#4497F2",
}

SYSTEMS = {
    "HgInSn": {
        "runs": {
            "ATR": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_008_step2501_base_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_010_step2501_baseline_random_e20_300K_100ps",
        },
    },
    "InSe": {
        "runs": {
            "ATR": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_004_frame1001_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
    },
    "MgSi": {
        "runs": {
            "ATR": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_004_frame1001_base_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
    },
}

METRICS = [
    ("epot_atom", "Potential energy per atom (eV/atom)"),
    ("ekin", "Kinetic energy (eV)"),
    ("fmax", "Maximum force (eV/Å)"),
    ("temperature", "Temperature (K)"),
]


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


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading trajectories")
    all_series = {
        system_name: {label: load_run(run_dir) for label, run_dir in system["runs"].items()}
        for system_name, system in SYSTEMS.items()
    }

    fig, axes = plt.subplots(
        nrows=len(SYSTEMS),
        ncols=len(METRICS),
        figsize=(28.8, 15.0),
        sharex=False,
        constrained_layout=True,
    )

    for row, (system_name, series) in enumerate(all_series.items()):
        for col, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for label in ("Baseline", "ATR"):
                data = series[label]
                ax.plot(data["time"], data[metric], color=COLORS[label], lw=1.1)
            if metric == "temperature":
                ax.axhline(300, color="#666666", lw=1.0, ls="--", alpha=0.75, zorder=0)
            ax.set_xlim(-4, 104)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel, fontsize=11)
            ax.text(
                0.02,
                0.92,
                system_name,
                transform=ax.transAxes,
                fontsize=11,
                ha="left",
                va="top",
            )

    png = OUT_DIR / "AllSystems_300K_ATR_vs_Baseline_12panel.png"
    pdf = OUT_DIR / "AllSystems_300K_ATR_vs_Baseline_12panel.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pil_kwargs={"compress_level": 1})
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {png}")
    print(f"wrote: {pdf}")


if __name__ == "__main__":
    main()
