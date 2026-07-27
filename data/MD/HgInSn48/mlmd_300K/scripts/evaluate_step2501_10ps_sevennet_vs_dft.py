#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
THIS_DIR = ROOT / "luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes"
META_DIR = ROOT / "luyouqi/meta_data"

if str(THIS_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(THIS_DIR / "scripts"))
if str(META_DIR) not in sys.path:
    sys.path.insert(0, str(META_DIR))

from evaluate_step2501_10ps_finetuned_baseline_vs_dft import (  # noqa: E402
    calc_metrics,
    load_dft_reference,
)
from sevennet_official_runner import (  # noqa: E402
    atoms_to_graph_data,
    build_calculators,
    resolve_accelerator,
    sevennet_predict,
    sevennet_predict_batch,
)


MODELS = {
    "7net-omni-i12-mp_r2scan": {
        "model": "7net-omni-i12",
        "file_type": None,
        "modal_candidates": ["mp r2scan", "mp_r2scan"],
    },
    "7net-omni-i12-matpes_r2scan": {
        "model": "7net-omni-i12",
        "file_type": None,
        "modal_candidates": ["matpes_r2scan"],
    },
}


def predict_batch_with_fallback(items, calc_gpu, calc_cpu, preferred_batch_size: int):
    e_pred = []
    f_pred = []
    stats = {"cuda": 0, "cpu": 0, "cpu_fallback": 0}

    if calc_gpu is None or preferred_batch_size <= 1:
        for item in items:
            e_tot, forces, backend = sevennet_predict(item["atoms"], calc_gpu, calc_cpu)
            e_pred.append(e_tot)
            f_pred.append(forces)
            stats[backend] = stats.get(backend, 0) + 1
        return e_pred, f_pred, stats

    start = 0
    chunk_size = min(preferred_batch_size, len(items))
    while start < len(items):
        end = min(start + chunk_size, len(items))
        chunk = items[start:end]
        try:
            batch_results = sevennet_predict_batch(chunk, calc_gpu, "cuda")
            for e_tot, forces, backend in batch_results:
                e_pred.append(e_tot)
                f_pred.append(forces)
                stats[backend] = stats.get(backend, 0) + 1
            start = end
            chunk_size = min(preferred_batch_size, len(items) - start) if start < len(items) else 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                continue
            e_tot, forces, backend = sevennet_predict(chunk[0]["atoms"], calc_gpu, calc_cpu)
            e_pred.append(e_tot)
            f_pred.append(forces)
            stats[backend] = stats.get(backend, 0) + 1
            start = end
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) and chunk_size > 1:
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                continue
            if chunk_size > 1:
                print(
                    f"[WARN] batch chunk_size={chunk_size} failed, retry smaller chunks: {exc}",
                    flush=True,
                )
                chunk_size = max(1, chunk_size // 2)
                continue
            e_tot, forces, backend = sevennet_predict(chunk[0]["atoms"], calc_gpu, calc_cpu)
            e_pred.append(e_tot)
            f_pred.append(forces)
            stats[backend] = stats.get(backend, 0) + 1
            start = end

    return e_pred, f_pred, stats


def evaluate_sevennet_model(
    label: str,
    settings: dict,
    atoms_ref: list,
    e_ref: np.ndarray,
    f_ref: np.ndarray,
    batch_size: int,
    accelerator_preference: str,
):
    selected_accel, accel_flags, availability = resolve_accelerator(accelerator_preference)
    print(
        "Loading SevenNet: "
        f"label={label}, model={settings['model']}, modal_candidates={settings['modal_candidates']}, "
        f"cuda={torch.cuda.is_available()}, selected_accelerator={selected_accel}, "
        f"accelerator_availability={availability}, batch_size={batch_size}",
        flush=True,
    )
    calc_gpu, calc_cpu, selected_modal = build_calculators(
        settings["model"],
        settings["file_type"],
        settings["modal_candidates"],
        accel_flags,
    )
    print(f"[INFO] {label}: selected_modal={selected_modal!r}", flush=True)

    graph_builder_calc = calc_gpu if calc_gpu is not None else calc_cpu
    pending = []
    e_pred = []
    f_pred = []
    stats = {"cuda": 0, "cpu": 0, "cpu_fallback": 0}
    n_frames = len(atoms_ref)

    for idx, atoms in enumerate(atoms_ref):
        atoms_copy = atoms.copy()
        graph = atoms_to_graph_data(atoms_copy, graph_builder_calc) if batch_size > 1 else None
        pending.append({"atoms": atoms_copy, "graph": graph})
        if len(pending) >= batch_size:
            e_chunk, f_chunk, chunk_stats = predict_batch_with_fallback(
                pending, calc_gpu, calc_cpu, batch_size
            )
            e_pred.extend(e_chunk)
            f_pred.extend(f_chunk)
            for key, value in chunk_stats.items():
                stats[key] = stats.get(key, 0) + value
            pending.clear()
            print(f"{label}: evaluated {len(e_pred)}/{n_frames} frames", flush=True)

    if pending:
        e_chunk, f_chunk, chunk_stats = predict_batch_with_fallback(
            pending, calc_gpu, calc_cpu, batch_size
        )
        e_pred.extend(e_chunk)
        f_pred.extend(f_chunk)
        for key, value in chunk_stats.items():
            stats[key] = stats.get(key, 0) + value
        print(f"{label}: evaluated {len(e_pred)}/{n_frames} frames", flush=True)

    e_pred_arr = np.array(e_pred, dtype=float)
    f_pred_arr = np.array(f_pred, dtype=float)
    metrics = calc_metrics(e_ref, f_ref, e_pred_arr, f_pred_arr)
    per_frame = np.stack(
        [
            np.arange(n_frames, dtype=float),
            np.arange(n_frames, dtype=float) * 0.001,
            e_ref / 48,
            e_pred_arr / 48,
            np.linalg.norm(f_ref, axis=2).max(axis=1),
            np.linalg.norm(f_pred_arr, axis=2).max(axis=1),
        ],
        axis=1,
    )
    return metrics, per_frame, selected_modal, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/eval_step2501_10ps_7net_omni_i12_vs_dft"),
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accelerator-preference", default="auto")
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else None
    atoms_ref, e_ref, f_ref, temp_ref = load_dft_reference(max_frames=max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"loaded DFT frames: {len(atoms_ref)}, "
        f"T_mean={float(np.mean(temp_ref)):.3f} K, T_min={float(np.min(temp_ref)):.1f} K, "
        f"T_max={float(np.max(temp_ref)):.1f} K",
        flush=True,
    )

    rows = []
    for label, settings in MODELS.items():
        metrics, per_frame, selected_modal, stats = evaluate_sevennet_model(
            label,
            settings,
            atoms_ref,
            e_ref,
            f_ref,
            args.batch_size,
            args.accelerator_preference,
        )
        row = {
            "label": label,
            "model": settings["model"],
            "selected_modal": selected_modal,
            "n_frames": len(atoms_ref),
            "time_ps": len(atoms_ref) * 0.001,
            "cuda_frames": stats.get("cuda", 0),
            "cpu_frames": stats.get("cpu", 0),
            "cpu_fallback_frames": stats.get("cpu_fallback", 0),
        }
        row.update(metrics)
        rows.append(row)

        with (args.out_dir / f"{label}_per_frame_energy_fmax.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "time_ps",
                    "dft_epot_eV_per_atom",
                    "pred_epot_eV_per_atom",
                    "dft_fmax_eV_per_A",
                    "pred_fmax_eV_per_A",
                ]
            )
            writer.writerows(per_frame)

    with (args.out_dir / "summary_mae.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out_dir / 'summary_mae.csv'}", flush=True)


if __name__ == "__main__":
    main()
