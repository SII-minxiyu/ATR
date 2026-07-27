#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import iread

ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
UTILS = ROOT / "teacher" / "teacher_structure"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from utils_chgnet import load_model, seed_everything  # noqa: E402


def evaluate(dataset: Path, checkpoint: str, device: str, stride: int, max_frames: int | None) -> dict[str, float]:
    from chgnet.model.dynamics import CHGNetCalculator

    model = load_model(checkpoint, use_device=device, verbose=True)
    model.eval()
    calc = CHGNetCalculator(model=model, use_device=device)

    e_ref = []
    e_pred = []
    f_ref = []
    f_pred = []
    n_seen = 0
    n_used = 0

    for frame_idx, atoms in enumerate(iread(str(dataset), index=":")):
        n_seen += 1
        if frame_idx % stride != 0:
            continue
        ref_e = atoms.get_potential_energy() / len(atoms)
        ref_f = atoms.get_forces()
        pred_atoms = atoms.copy()
        pred_atoms.calc = calc
        pred_e = pred_atoms.get_potential_energy() / len(pred_atoms)
        pred_f = pred_atoms.get_forces()

        e_ref.append(ref_e)
        e_pred.append(pred_e)
        f_ref.append(ref_f)
        f_pred.append(pred_f)
        n_used += 1
        if max_frames is not None and n_used >= max_frames:
            break

    e_ref_arr = np.array(e_ref)
    e_pred_arr = np.array(e_pred)
    f_ref_arr = np.concatenate(f_ref, axis=0)
    f_pred_arr = np.concatenate(f_pred, axis=0)
    f_ref_norm = np.linalg.norm(f_ref_arr, axis=1)
    f_pred_norm = np.linalg.norm(f_pred_arr, axis=1)
    f_vec_err = np.linalg.norm(f_pred_arr - f_ref_arr, axis=1)

    return {
        "n_seen": float(n_seen),
        "n_eval_frames": float(n_used),
        "energy_mae_eV_per_atom": float(np.mean(np.abs(e_pred_arr - e_ref_arr))),
        "energy_rmse_eV_per_atom": float(np.sqrt(np.mean((e_pred_arr - e_ref_arr) ** 2))),
        "energy_bias_eV_per_atom": float(np.mean(e_pred_arr - e_ref_arr)),
        "force_component_mae_eV_per_A": float(np.mean(np.abs(f_pred_arr - f_ref_arr))),
        "force_component_rmse_eV_per_A": float(np.sqrt(np.mean((f_pred_arr - f_ref_arr) ** 2))),
        "force_vector_mae_eV_per_A": float(np.mean(f_vec_err)),
        "force_vector_rmse_eV_per_A": float(np.sqrt(np.mean(f_vec_err**2))),
        "force_norm_mae_eV_per_A": float(np.mean(np.abs(f_pred_norm - f_ref_norm))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    seed_everything(20260514)
    models = {
        "finetuned_balanced": "/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar",
        "official_r2scan": "r2scan",
        "baseline_random_e20": "/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/baseline_default_lmdb_random_ddp4_e20/best.pth.tar",
    }

    rows = []
    max_frames = args.max_frames if args.max_frames > 0 else None
    for label, checkpoint in models.items():
        print(f"Evaluating {label} ...", flush=True)
        row = {"label": label, "checkpoint": checkpoint, "stride": args.stride}
        row.update(evaluate(args.dataset, checkpoint, args.device, args.stride, max_frames))
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
