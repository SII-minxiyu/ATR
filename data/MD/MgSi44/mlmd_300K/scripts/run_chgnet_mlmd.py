#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path("/inspire/hdd/global_user/luomingxiang-240108540155")
UTILS = ROOT / "teacher" / "teacher_structure"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from utils_chgnet import load_model, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True, help="Initial structure extxyz")
    parser.add_argument("--checkpoint", required=True, help="CHGNet checkpoint path")
    parser.add_argument("--out-dir", required=True, help="Output run directory")
    parser.add_argument("--steps", type=int, default=10_000, help="MD steps")
    parser.add_argument("--temperature", type=float, default=300.0, help="Target temperature in K")
    parser.add_argument("--starting-temperature", type=float, default=300.0, help="Initial velocity temperature in K")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs")
    parser.add_argument("--loginterval", type=int, default=10, help="ASE log/trajectory interval")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--thermostat", default="Berendsen", help="Berendsen or Nose-Hoover")
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        from ase.io import read, write
        from chgnet.model.dynamics import MolecularDynamics
    except Exception as exc:
        raise SystemExit(
            "Failed to import ML-MD dependencies. The Python environment must provide "
            "torch, ase, pymatgen, and chgnet.\n"
            f"Original error: {exc}"
        ) from exc

    initial = Path(args.initial).resolve()
    pretrained_names = {"0.3.0", "0.2.0", "r2scan"}
    checkpoint = args.checkpoint if args.checkpoint in pretrained_names else str(Path(args.checkpoint).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(initial), index=0)
    model = load_model(str(checkpoint), use_device=args.device, verbose=True)
    model.eval()

    trajectory_path = out_dir / "mlmd.traj"
    log_path = out_dir / "mlmd.log"
    summary_path = out_dir / "run_config.json"

    config = {
        "initial": str(initial),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "steps": args.steps,
        "temperature_K": args.temperature,
        "starting_temperature_K": args.starting_temperature,
        "timestep_fs": args.timestep,
        "loginterval": args.loginterval,
        "device": args.device,
        "thermostat": args.thermostat,
        "seed": args.seed,
        "natoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }
    summary_path.write_text(json.dumps(config, indent=2) + "\n")

    md = MolecularDynamics(
        atoms=atoms,
        model=model,
        ensemble="nvt",
        thermostat=args.thermostat,
        temperature=args.temperature,
        starting_temperature=args.starting_temperature,
        timestep=args.timestep,
        trajectory=str(trajectory_path),
        logfile=str(log_path),
        loginterval=args.loginterval,
        use_device=args.device,
    )
    md.run(args.steps)

    frames = read(str(trajectory_path), index=":")
    extxyz_path = out_dir / "mlmd_trajectory.extxyz"
    write(str(extxyz_path), frames, format="extxyz")
    write(str(out_dir / "final_structure.extxyz"), frames[-1], format="extxyz")
    print(json.dumps({**config, "trajectory": str(trajectory_path), "extxyz": str(extxyz_path)}, indent=2))


if __name__ == "__main__":
    main()
