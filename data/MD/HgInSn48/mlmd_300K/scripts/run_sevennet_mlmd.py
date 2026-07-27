#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
META_DIR = ROOT / "luyouqi/meta_data"
if str(META_DIR) not in sys.path:
    sys.path.insert(0, str(META_DIR))

from sevennet_official_runner import build_calculators, resolve_accelerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True, help="Initial structure extxyz")
    parser.add_argument("--out-dir", required=True, help="Output run directory")
    parser.add_argument("--model-label", required=True, help="Human readable model label")
    parser.add_argument("--sevennet-model", default="7net-omni-i12")
    parser.add_argument("--modal-candidates", nargs="+", required=True)
    parser.add_argument("--file-type", default=None)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--starting-temperature", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs")
    parser.add_argument("--loginterval", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--thermostat", default="Berendsen")
    parser.add_argument("--taut-fs", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--accelerator-preference", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10000)
    return parser.parse_args()


def ensure_output_dir(out_dir: Path, allow_existing: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_existing:
        raise SystemExit(f"Output directory exists and is non-empty, refusing to overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def write_log_header(handle) -> None:
    handle.write("Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n")
    handle.flush()


def log_state(handle, atoms, step: int, timestep_fs: float) -> None:
    epot = float(atoms.get_potential_energy())
    ekin = float(atoms.get_kinetic_energy())
    temp = float(atoms.get_temperature())
    time_ps = step * timestep_fs * 0.001
    handle.write(f"{time_ps:7.4f} {epot + ekin:15.4f} {epot:12.4f} {ekin:12.4f} {temp:7.1f}\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        import torch
        from ase import units
        from ase.io import read, write
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        from ase.io.trajectory import Trajectory
    except Exception as exc:
        raise SystemExit(f"Failed to import ML-MD dependencies: {exc}") from exc

    initial = Path(args.initial).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_output_dir(out_dir, args.allow_existing)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested device=cuda but torch.cuda.is_available() is False")

    atoms = read(str(initial), index=0)
    selected_accel, accel_flags, availability = resolve_accelerator(args.accelerator_preference)
    calc_gpu, calc_cpu, selected_modal = build_calculators(
        args.sevennet_model,
        args.file_type,
        args.modal_candidates,
        accel_flags,
    )
    if args.device == "cuda":
        if calc_gpu is None:
            raise SystemExit("Failed to build CUDA SevenNet calculator")
        atoms.calc = calc_gpu
    else:
        atoms.calc = calc_cpu

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=args.starting_temperature,
        force_temp=True,
        rng=np.random,
    )
    Stationary(atoms)
    ZeroRotation(atoms)

    trajectory_path = out_dir / "mlmd.traj"
    log_path = out_dir / "mlmd.log"
    config_path = out_dir / "run_config.json"
    config = {
        "initial": str(initial),
        "out_dir": str(out_dir),
        "model_label": args.model_label,
        "sevennet_model": args.sevennet_model,
        "selected_modal": selected_modal,
        "modal_candidates": args.modal_candidates,
        "file_type": args.file_type,
        "steps": args.steps,
        "temperature_K": args.temperature,
        "starting_temperature_K": args.starting_temperature,
        "ensemble": "nvt" if args.thermostat.lower() == "berendsen" else "nve",
        "thermostat": args.thermostat,
        "taut_fs": args.taut_fs,
        "timestep_fs": args.timestep,
        "loginterval": args.loginterval,
        "device": args.device,
        "seed": args.seed,
        "accelerator_preference": args.accelerator_preference,
        "selected_accelerator": selected_accel,
        "accelerator_availability": availability,
        "natoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    timestep = args.timestep * units.fs
    if args.thermostat.lower() == "berendsen":
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep,
            temperature_K=args.temperature,
            taut=args.taut_fs * units.fs,
            fixcm=True,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=timestep)

    traj = Trajectory(str(trajectory_path), "w", atoms)
    with log_path.open("w", buffering=1) as log_handle:
        write_log_header(log_handle)
        last_recorded_step = {"value": 0}
        log_state(log_handle, atoms, 0, args.timestep)
        traj.write(atoms)

        def record() -> None:
            step = dyn.nsteps
            if step == last_recorded_step["value"]:
                return
            log_state(log_handle, atoms, step, args.timestep)
            traj.write(atoms)
            last_recorded_step["value"] = step
            if args.progress_interval > 0 and step % args.progress_interval == 0:
                print(f"{args.model_label}: completed {step}/ {args.steps} steps ({step * args.timestep * 0.001:.1f} ps)", flush=True)

        dyn.attach(record, interval=args.loginterval)
        dyn.run(args.steps)
    traj.close()

    frames = read(str(trajectory_path), index=":")
    extxyz_path = out_dir / "mlmd_trajectory.extxyz"
    write(str(extxyz_path), frames, format="extxyz")
    write(str(out_dir / "final_structure.extxyz"), frames[-1], format="extxyz")
    print(json.dumps({**config, "trajectory": str(trajectory_path), "extxyz": str(extxyz_path)}, indent=2))


if __name__ == "__main__":
    main()
