from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import iread


def _attach_calc_if_available(atoms) -> None:
    calc_kwargs = {}

    energy = atoms.info.get("energy")
    if energy is not None:
        try:
            energy = float(energy)
            if np.isfinite(energy):
                calc_kwargs["energy"] = energy
        except (TypeError, ValueError):
            pass

    free_energy = atoms.info.get("free_energy")
    if free_energy is not None:
        try:
            free_energy = float(free_energy)
            if np.isfinite(free_energy):
                calc_kwargs["free_energy"] = free_energy
        except (TypeError, ValueError):
            pass

    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=float)
        if np.all(np.isfinite(forces)):
            calc_kwargs["forces"] = forces

    stress = atoms.info.get("stress")
    if stress is not None:
        stress = np.asarray(stress, dtype=float).reshape(-1)
        if stress.size in (6, 9) and np.all(np.isfinite(stress)):
            calc_kwargs["stress"] = stress

    if calc_kwargs:
        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ExtXYZ to ASE db for router inference.")
    parser.add_argument("--xyz", type=Path, required=True, help="Input .xyz/.extxyz file.")
    parser.add_argument("--out-db", type=Path, required=True, help="Output ASE .db path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output db if it exists.")
    args = parser.parse_args()

    if args.out_db.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_db} already exists. Use --overwrite to replace it.")
        args.out_db.unlink()

    args.out_db.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with connect(args.out_db) as db:
        for atoms in iread(args.xyz, index=":"):
            _attach_calc_if_available(atoms)
            key_value_pairs = {}
            structure_id = atoms.info.get("id")
            immutable_id = atoms.info.get("immutable_id")
            if structure_id is not None:
                key_value_pairs["ID"] = str(structure_id)
            if immutable_id is not None:
                key_value_pairs["IMMUTABLE_ID"] = str(immutable_id)
            db.write(atoms, key_value_pairs=key_value_pairs)
            written += 1

    print(f"Wrote {written} structures to {args.out_db}")


if __name__ == "__main__":
    main()
