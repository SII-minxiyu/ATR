from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write


def _parse_teacher_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Each --teacher must be given as MODEL_NAME=/path/to/file.extxyz"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise argparse.ArgumentTypeError("Teacher model name cannot be empty.")
    return name, path


def _extract_energy(atoms) -> float | None:
    energy = atoms.info.get("energy")
    if energy is None and getattr(atoms, "calc", None) is not None:
        energy = getattr(atoms.calc, "results", {}).get("energy")
    if energy is None:
        return None
    try:
        energy = float(energy)
    except Exception:
        return None
    return energy if np.isfinite(energy) else None


def _extract_forces(atoms) -> np.ndarray | None:
    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=float)
        if np.all(np.isfinite(forces)):
            return forces
        return None
    if getattr(atoms, "calc", None) is not None:
        forces = getattr(atoms.calc, "results", {}).get("forces")
        if forces is not None:
            forces = np.asarray(forces, dtype=float)
            if np.all(np.isfinite(forces)):
                return forces
    return None


def _label_from_atoms(atoms) -> tuple[str | None, str | None]:
    sid = atoms.info.get("id") or atoms.info.get("ID")
    immutable = atoms.info.get("immutable_id") or atoms.info.get("IMMUTABLE_ID")
    sid = None if sid is None else str(sid)
    immutable = None if immutable is None else str(immutable)
    return sid, immutable


def _compare_structure(anchor, other, *, idx: int, other_name: str) -> None:
    if len(anchor) != len(other):
        raise ValueError(f"Structure {idx}: natoms mismatch for {other_name}")
    if not np.array_equal(anchor.numbers, other.numbers):
        raise ValueError(f"Structure {idx}: atomic numbers mismatch for {other_name}")
    anchor_id, anchor_immutable = _label_from_atoms(anchor)
    other_id, other_immutable = _label_from_atoms(other)
    if anchor_id is not None and other_id is not None and anchor_id != other_id:
        raise ValueError(f"Structure {idx}: id mismatch for {other_name}: {anchor_id} vs {other_id}")
    if (
        anchor_immutable is not None
        and other_immutable is not None
        and anchor_immutable != other_immutable
    ):
        raise ValueError(
            f"Structure {idx}: immutable_id mismatch for {other_name}: {anchor_immutable} vs {other_immutable}"
        )


def _iter_aligned_teacher_atoms(teacher_specs: list[tuple[str, Path]]):
    iterators = [(name, iread(path, index=":", format="extxyz")) for name, path in teacher_specs]
    idx = 0
    while True:
        rows = []
        ended = []
        for name, iterator in iterators:
            try:
                atoms = next(iterator)
            except StopIteration:
                ended.append(name)
                atoms = None
            rows.append((name, atoms))

        if ended:
            if len(ended) != len(iterators):
                raise RuntimeError(f"Teacher files ended inconsistently at structure {idx + 1}: {ended}")
            break

        idx += 1
        anchor_name, anchor_atoms = rows[0]
        teacher_atoms = {anchor_name: anchor_atoms}
        for name, atoms in rows[1:]:
            _compare_structure(anchor_atoms, atoms, idx=idx, other_name=name)
            teacher_atoms[name] = atoms
        yield idx, anchor_atoms, teacher_atoms


def _attach_prediction_calc(atoms, source_atoms=None) -> None:
    calc_kwargs = {}
    source = atoms if source_atoms is None else source_atoms
    energy = _extract_energy(source)
    if energy is not None:
        calc_kwargs["energy"] = energy
    forces = _extract_forces(source)
    if forces is not None:
        calc_kwargs["forces"] = forces
    if calc_kwargs:
        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export accepted extxyz with chosen teacher energy/forces from an existing route decisions CSV."
    )
    parser.add_argument(
        "--teacher",
        action="append",
        type=_parse_teacher_arg,
        required=True,
        help="Repeat as MODEL_NAME=/path/to/teacher.extxyz",
    )
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-accepted-xyz", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    if "structure_id" not in decisions.columns or "status" not in decisions.columns:
        raise SystemExit("decisions CSV must contain at least structure_id and status columns")

    keep = decisions[decisions["status"] == "accept"].copy()
    keep["structure_id"] = keep["structure_id"].astype(int)
    decision_map = {int(row["structure_id"]): row for _, row in keep.iterrows()}

    ordered_specs = args.teacher
    args.output_accepted_xyz.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.output_accepted_xyz, "w", encoding="utf-8") as handle:
        for idx, _anchor_atoms, teacher_atoms in _iter_aligned_teacher_atoms(ordered_specs):
            row = decision_map.get(idx)
            if row is None:
                if args.log_every > 0 and idx % args.log_every == 0:
                    print(f"[export-accepted] scanned {idx} structures; written {written}")
                continue

            chosen_model = str(row["chosen_model"])
            if chosen_model not in teacher_atoms:
                raise KeyError(f"Chosen model {chosen_model!r} not found among provided teachers at structure {idx}")

            source_atoms = teacher_atoms[chosen_model]
            atoms = source_atoms.copy()
            _attach_prediction_calc(atoms, source_atoms=source_atoms)
            atoms.info.update(
                {
                    "route_status": str(row.get("status", "")),
                    "route_chosen_model": chosen_model,
                    "route_chosen_label": str(row.get("chosen_label", "")),
                    "route_confidence": float(row.get("confidence", float("nan"))),
                    "route_postprocess_reject": bool(row.get("postprocess_reject", False)),
                    "route_postprocess_reason": "" if pd.isna(row.get("postprocess_reason")) else str(row.get("postprocess_reason")),
                }
            )
            write(handle, atoms, format="extxyz")
            written += 1

            if args.log_every > 0 and idx % args.log_every == 0:
                print(f"[export-accepted] scanned {idx} structures; written {written}")

    summary = {
        "decisions_csv": str(args.decisions_csv),
        "output_accepted_xyz": str(args.output_accepted_xyz),
        "teacher_files": {name: str(path) for name, path in ordered_specs},
        "accepted_rows_in_csv": int(len(keep)),
        "written_structures": int(written),
    }
    summary_path = args.output_summary_json or args.output_accepted_xyz.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
