#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Production extxyz trajectory")
    parser.add_argument("--out-dir", required=True, help="Directory for selected starts")
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="source_step values to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read(args.input, index=":")
    by_step = {int(atoms.info["source_step"]): atoms for atoms in frames}
    rows = []

    for step in args.steps:
        if step not in by_step:
            raise SystemExit(f"source_step {step} not found in {args.input}")
        atoms = by_step[step]
        base = f"HgInSn48_source_step_{step}"
        extxyz_path = out_dir / f"{base}.extxyz"
        poscar_path = out_dir / f"{base}.vasp"
        write(extxyz_path, atoms, format="extxyz")
        write(poscar_path, atoms, format="vasp", direct=True, sort=False, vasp5=True)
        rows.append(
            {
                "source_step": step,
                "source_time_fs": step,
                "source_time_ps": step / 1000.0,
                "formula": atoms.get_chemical_formula(),
                "natoms": len(atoms),
                "energy_eV": float(atoms.get_potential_energy()),
                "extxyz": str(extxyz_path),
                "poscar": str(poscar_path),
            }
        )

    summary_path = out_dir / "selected_start_frames.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
