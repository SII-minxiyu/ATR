#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
UTILS = ROOT / "teacher" / "teacher_structure"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from utils_chgnet import load_model, seed_everything  # noqa: E402


NATOMS = 48
DFT_ROOT = ROOT / "luyouqi/md_HgInSn_48atom_vasp/reference_dft_md"
DFT_SEGMENTS = [
    # source_step=2501 aligned 10 ps reference: run_004 index 500..end plus run_006 all frames.
    (DFT_ROOT / "run_004_r2scan_nvt_300K_extend_3ps", 500, None),
    (DFT_ROOT / "run_006_r2scan_nvt_300K_extend_7p5ps", 0, None),
]
MODELS = {
    "finetuned_balanced": ROOT
    / "teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar",
    "baseline_random_e20": ROOT / "teacher/chgnet_distill_pretrain/baseline_default_lmdb_random_ddp4_e20/best.pth.tar",
}


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


def parse_outcar_forces(path: Path, natoms: int = NATOMS) -> np.ndarray:
    frames = []
    with path.open(errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            if "TOTAL-FORCE (eV/Angst)" not in line:
                continue
            next(lines, None)
            forces = []
            for _ in range(natoms):
                parts = next(lines).split()
                if len(parts) >= 6:
                    forces.append([float(x) for x in parts[-3:]])
            if len(forces) == natoms:
                frames.append(forces)
    if not frames:
        raise RuntimeError(f"No force blocks parsed from {path}")
    return np.array(frames, dtype=float)


def load_dft_reference(max_frames: int | None = None) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    all_atoms = []
    all_epot = []
    all_forces = []
    all_temp = []
    for seg_dir, start, stop in DFT_SEGMENTS:
        osz = parse_oszicar(seg_dir / "OSZICAR")
        forces = parse_outcar_forces(seg_dir / "OUTCAR")
        atoms = read(seg_dir / "XDATCAR", ":")
        n = min(len(atoms), len(osz["Epot"]), len(forces))
        sl = slice(start, stop if stop is not None else n)
        all_atoms.extend(atoms[:n][sl])
        all_epot.append(osz["Epot"][:n][sl])
        all_forces.append(forces[:n][sl])
        all_temp.append(osz["T_K"][:n][sl])

    atoms_ref = all_atoms
    epot_ref = np.concatenate(all_epot)
    force_ref = np.concatenate(all_forces, axis=0)
    temp_ref = np.concatenate(all_temp)
    if max_frames is not None:
        atoms_ref = atoms_ref[:max_frames]
        epot_ref = epot_ref[:max_frames]
        force_ref = force_ref[:max_frames]
        temp_ref = temp_ref[:max_frames]
    return atoms_ref, epot_ref, force_ref, temp_ref


def calc_metrics(e_ref_total: np.ndarray, f_ref: np.ndarray, e_pred_total: np.ndarray, f_pred: np.ndarray) -> dict[str, float]:
    e_ref_atom = e_ref_total / NATOMS
    e_pred_atom = e_pred_total / NATOMS
    e_err_atom = e_pred_atom - e_ref_atom
    f_err = f_pred - f_ref
    f_vec_err = np.linalg.norm(f_err, axis=2)
    f_ref_norm = np.linalg.norm(f_ref, axis=2)
    f_pred_norm = np.linalg.norm(f_pred, axis=2)
    fmax_ref = f_ref_norm.max(axis=1)
    fmax_pred = f_pred_norm.max(axis=1)
    fmax_err = fmax_pred - fmax_ref
    return {
        "energy_mae_eV_per_atom": float(np.mean(np.abs(e_err_atom))),
        "energy_rmse_eV_per_atom": float(np.sqrt(np.mean(e_err_atom**2))),
        "energy_bias_eV_per_atom": float(np.mean(e_err_atom)),
        "force_component_mae_eV_per_A": float(np.mean(np.abs(f_err))),
        "force_component_rmse_eV_per_A": float(np.sqrt(np.mean(f_err**2))),
        "force_vector_mae_eV_per_A": float(np.mean(f_vec_err)),
        "force_vector_rmse_eV_per_A": float(np.sqrt(np.mean(f_vec_err**2))),
        "force_norm_mae_eV_per_A": float(np.mean(np.abs(f_pred_norm - f_ref_norm))),
        "fmax_mae_eV_per_A": float(np.mean(np.abs(fmax_err))),
        "fmax_rmse_eV_per_A": float(np.sqrt(np.mean(fmax_err**2))),
        "fmax_bias_eV_per_A": float(np.mean(fmax_err)),
        "fmax_ref_mean_eV_per_A": float(np.mean(fmax_ref)),
        "fmax_pred_mean_eV_per_A": float(np.mean(fmax_pred)),
        "fmax_ref_max_eV_per_A": float(np.max(fmax_ref)),
        "fmax_pred_max_eV_per_A": float(np.max(fmax_pred)),
    }


def evaluate_model(label: str, checkpoint: Path | str, atoms_ref: list, e_ref: np.ndarray, f_ref: np.ndarray, device: str) -> tuple[dict[str, float], np.ndarray]:
    from chgnet.model.dynamics import CHGNetCalculator

    model = load_model(str(checkpoint), use_device=device, verbose=True)
    model.eval()
    calc = CHGNetCalculator(model=model, use_device=device)

    e_pred = []
    f_pred = []
    for idx, atoms in enumerate(atoms_ref):
        pred_atoms = atoms.copy()
        pred_atoms.calc = calc
        e_pred.append(float(pred_atoms.get_potential_energy()))
        f_pred.append(pred_atoms.get_forces())
        if (idx + 1) % 500 == 0:
            print(f"{label}: evaluated {idx + 1}/{len(atoms_ref)} frames", flush=True)

    e_pred_arr = np.array(e_pred, dtype=float)
    f_pred_arr = np.array(f_pred, dtype=float)
    return calc_metrics(e_ref, f_ref, e_pred_arr, f_pred_arr), np.stack(
        [
            np.arange(len(atoms_ref), dtype=float),
            np.arange(len(atoms_ref), dtype=float) * 0.001,
            e_ref / NATOMS,
            e_pred_arr / NATOMS,
            np.linalg.norm(f_ref, axis=2).max(axis=1),
            np.linalg.norm(f_pred_arr, axis=2).max(axis=1),
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eval_step2501_10ps_finetuned_vs_baseline_dft"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    seed_everything(20260514)
    max_frames = args.max_frames if args.max_frames > 0 else None
    atoms_ref, e_ref, f_ref, temp_ref = load_dft_reference(max_frames=max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded DFT frames: {len(atoms_ref)}", flush=True)

    rows = []
    for label, checkpoint in MODELS.items():
        print(f"Evaluating {label}", flush=True)
        metrics, per_frame = evaluate_model(label, checkpoint, atoms_ref, e_ref, f_ref, args.device)
        row = {
            "label": label,
            "checkpoint": str(checkpoint),
            "n_frames": len(atoms_ref),
            "time_ps": len(atoms_ref) * 0.001,
        }
        row.update(metrics)
        rows.append(row)
        with (args.out_dir / f"{label}_per_frame_energy_fmax.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "time_ps", "dft_epot_eV_per_atom", "pred_epot_eV_per_atom", "dft_fmax_eV_per_A", "pred_fmax_eV_per_A"])
            writer.writerows(per_frame)

    with (args.out_dir / "summary_mae.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out_dir / 'summary_mae.csv'}", flush=True)


if __name__ == "__main__":
    main()
