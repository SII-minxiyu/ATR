#!/usr/bin/env python
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread, read


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
OUT_DIR = ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/picture"
KB_EV_PER_K = 8.617333262145e-5

COLORS = {
    "AIMD": "#111111",
    "ATR": "#FF8E00",
    "Teacher": "#FF6445",
    "Baseline": "#4497F2",
}


SYSTEMS = {
    "HgInSn": {
        "natoms": 48,
        "target_T": 300,
        "mlmd": {
            "ATR": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_008_step2501_base_300K_100ps",
            "Teacher": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_009_step2501_official_r2scan_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_010_step2501_baseline_random_e20_300K_100ps",
        },
        "aimd_segments": [
            # source_step=2501: run_004 index 500 to end plus run_006 all frames.
            (ROOT / "luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps", 500, None),
            (ROOT / "luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_006_r2scan_nvt_300K_extend_7p5ps", 0, None),
        ],
    },
    "InSe": {
        "natoms": 48,
        "target_T": 300,
        "mlmd": {
            "ATR": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_004_frame1001_300K_100ps",
            "Teacher": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_005_official_r2scan_frame1001_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
        "aimd_extxyz": ROOT / "luyouqi/md_InSe_48atom_vasp/ml_test_dataset_r2scan_InSe48/r2scan_300K/InSe48_r2scan_300K_2ps_all.extxyz",
        "aimd_start_index": 1000,
    },
    "MgSi": {
        "natoms": 44,
        "target_T": 300,
        "mlmd": {
            "ATR": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_004_frame1001_base_300K_100ps",
            "Teacher": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_005_frame1001_official_r2scan_300K_100ps",
            "Baseline": ROOT / "luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes/runs/run_006_baseline_random_e20_frame1001_300K_100ps",
        },
        "aimd_extxyz": ROOT / "luyouqi/md_MgSi_44atom_vasp/ml_test_dataset_r2scan_MgSi44/r2scan_300K/MgSi44_r2scan_300K_2ps_all.extxyz",
        "aimd_start_index": 1000,
    },
}


def parse_mlmd_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"No rows found in {run_dir / 'mlmd.log'}")
    return np.array(rows, dtype=float)


def fmax_series(frames) -> np.ndarray:
    values = []
    for atoms in frames:
        forces = atoms.get_forces()
        values.append(float(np.linalg.norm(forces, axis=1).max()))
    return np.array(values, dtype=float)


def load_mlmd(run_dir: Path) -> dict[str, np.ndarray]:
    log = parse_mlmd_log(run_dir)
    frames = read(run_dir / "mlmd_trajectory.extxyz", ":")
    if len(log) != len(frames):
        raise RuntimeError(f"{run_dir}: {len(log)} log rows vs {len(frames)} frames")
    natoms = len(frames[0])
    return {
        "time": log[:, 0],
        "temperature": log[:, 4],
        "epot_atom": log[:, 2] / natoms,
        "ekin": log[:, 3],
        "fmax": fmax_series(frames),
    }


def parse_oszicar(path: Path) -> dict[str, np.ndarray]:
    pattern = re.compile(
        r"^\s*\d+\s+T=\s*([+-]?\d+(?:\.\d*)?)\.\s+E=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
        r"\s+F=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
        r".*?EK=\s*([+-]?\.\d+E[+-]\d+|[+-]?\d+\.\d+E[+-]\d+|[+-]?\d+(?:\.\d*)?)"
    )
    temp, epot, ekin = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            t, _e, f, ek = match.groups()
            temp.append(float(t))
            epot.append(float(f))
            ekin.append(float(ek))
    if not temp:
        raise RuntimeError(f"No MD rows parsed from {path}")
    return {"temperature": np.array(temp), "epot": np.array(epot), "ekin": np.array(ekin)}


def parse_outcar_fmax(path: Path, natoms: int) -> np.ndarray:
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
    return np.array(vals)


def load_hginsn_aimd(system: dict) -> dict[str, np.ndarray]:
    pieces = {"temperature": [], "epot": [], "ekin": [], "fmax": []}
    natoms = system["natoms"]
    for seg_dir, start, stop in system["aimd_segments"]:
        osz = parse_oszicar(seg_dir / "OSZICAR")
        fmax = parse_outcar_fmax(seg_dir / "OUTCAR", natoms=natoms)
        n = min(len(osz["temperature"]), len(fmax))
        sl = slice(start, stop if stop is not None else n)
        pieces["temperature"].append(osz["temperature"][:n][sl])
        pieces["epot"].append(osz["epot"][:n][sl])
        pieces["ekin"].append(osz["ekin"][:n][sl])
        pieces["fmax"].append(fmax[:n][sl])
    data = {key: np.concatenate(vals) for key, vals in pieces.items()}
    n = len(data["temperature"])
    return {
        "time": np.arange(n, dtype=float) * 0.001,
        "temperature": data["temperature"],
        "epot_atom": data["epot"] / natoms,
        "ekin": data["ekin"],
        "fmax": data["fmax"],
    }


def load_extxyz_aimd(system: dict) -> dict[str, np.ndarray]:
    path = system["aimd_extxyz"]
    start = system["aimd_start_index"]
    natoms = system["natoms"]
    frames = list(iread(str(path), index=slice(start, None, None)))
    epot = np.array([atoms.get_potential_energy() / len(atoms) for atoms in frames], dtype=float)
    fmax = fmax_series(frames)
    temp = []
    for atoms in frames:
        val = atoms.info.get("temperature_K", atoms.info.get("temperature", np.nan))
        temp.append(float(val) if val is not None else np.nan)
    temp = np.array(temp, dtype=float)
    if not np.isfinite(temp).any():
        temp = np.full(len(frames), system["target_T"], dtype=float)
    else:
        temp = np.where(np.isfinite(temp), temp, system["target_T"])
    ekin = 1.5 * natoms * KB_EV_PER_K * temp
    return {
        "time": np.arange(len(frames), dtype=float) * 0.001,
        "temperature": temp,
        "epot_atom": epot,
        "ekin": ekin,
        "fmax": fmax,
    }


def load_aimd(system_name: str, system: dict) -> dict[str, np.ndarray]:
    if system_name == "HgInSn":
        return load_hginsn_aimd(system)
    return load_extxyz_aimd(system)


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
    draw_order = ["Baseline", "Teacher", "ATR", "AIMD"]
    for label in draw_order:
        data = series[label]
        is_aimd = label == "AIMD"
        ax.plot(
            data["time"],
            data[metric],
            color=COLORS[label],
            lw=1.45 if is_aimd else 1.1,
            zorder=4 if is_aimd else (3 if label == "ATR" else 2),
        )
    if metric == "temperature":
        ax.axhline(target_T, color="#666666", lw=1.0, ls="--", alpha=0.75, zorder=0)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    png = OUT_DIR / f"{system_name}_300K_ATR_Teacher_Baseline_AIMD_{metric}.png"
    pdf = OUT_DIR / f"{system_name}_300K_ATR_Teacher_Baseline_AIMD_{metric}.pdf"
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
        series = {"AIMD": load_aimd(system_name, system)}
        for label, run_dir in system["mlmd"].items():
            series[label] = load_mlmd(run_dir)
        for metric, ylabel in specs.items():
            plot_metric(system_name, system["target_T"], series, metric, ylabel)
    print(f"wrote figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
