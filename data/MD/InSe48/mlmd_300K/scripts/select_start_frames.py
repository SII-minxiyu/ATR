#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read, write


def plain(value):
    return value.item() if hasattr(value, "item") else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input extxyz trajectory")
    parser.add_argument("--out-dir", required=True, help="Directory for selected starts")
    parser.add_argument("--prefix", required=True, help="Filename prefix")
    parser.add_argument("--frames", nargs="+", type=int, required=True, help="1-based global frame numbers to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read(args.input, index=":")
    rows = []
    for frame_no in args.frames:
        if frame_no < 1 or frame_no > len(frames):
            raise SystemExit(f"frame {frame_no} outside 1..{len(frames)} for {args.input}")
        atoms = frames[frame_no - 1]
        atoms.info["global_frame"] = frame_no
        atoms.info["global_time_fs"] = frame_no
        base = f"{args.prefix}_frame_{frame_no}"
        extxyz_path = out_dir / f"{base}.extxyz"
        poscar_path = out_dir / f"{base}.vasp"
        write(extxyz_path, atoms, format="extxyz")
        write(poscar_path, atoms, format="vasp", direct=True, sort=False, vasp5=True)
        rows.append(
            {
                "global_frame": frame_no,
                "global_time_fs": frame_no,
                "global_time_ps": frame_no / 1000.0,
                "original_source_step": plain(atoms.info.get("source_step")),
                "formula": atoms.get_chemical_formula(),
                "natoms": len(atoms),
                "energy_eV": float(atoms.get_potential_energy()),
                "extxyz": str(extxyz_path),
                "poscar": str(poscar_path),
            }
        )

    (out_dir / "selected_start_frames.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
