from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from upet.calculator import UPETCalculator


def _safe_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return {}


def _extract_metadata(row) -> tuple[dict, dict]:
    data = _safe_dict(getattr(row, "data", None))
    kv = _safe_dict(getattr(row, "key_value_pairs", None))

    out_kv: dict[str, str] = {}
    immutable_id = (
        data.get("immutable_id")
        or kv.get("IMMUTABLE_ID")
        or kv.get("immutable_id")
    )
    sample_id = (
        data.get("id")
        or kv.get("ID")
        or kv.get("id")
    )

    if immutable_id is not None:
        out_kv["IMMUTABLE_ID"] = str(immutable_id)
    if sample_id is not None:
        out_kv["ID"] = str(sample_id)

    out_data = {}
    if sample_id is not None:
        out_data["id"] = str(sample_id)
    if immutable_id is not None:
        out_data["immutable_id"] = str(immutable_id)
    out_data["source_db_row_id"] = int(row.id)
    return out_kv, out_data


def _clean_atoms(row):
    try:
        atoms = row.toatoms(add_additional_information=True)
    except TypeError:
        atoms = row.toatoms()

    atoms.calc = None

    for key in ("energy", "free_energy", "stress"):
        atoms.info.pop(key, None)
    for key in ("forces", "stress"):
        if key in atoms.arrays:
            del atoms.arrays[key]
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PET-MAD on an ASE db and write predictions to a new ASE db."
    )
    parser.add_argument(
        "--input-db",
        type=Path,
        required=True,
        help="Input ASE .db containing the source structures.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        required=True,
        help="Output ASE .db for PET-MAD predictions.",
    )
    parser.add_argument(
        "--model",
        default="pet-mad-s",
        help="UPET model name, e.g. pet-mad-s / pet-mad-m / pet-mad-l.",
    )
    parser.add_argument(
        "--version",
        default="1.5.0",
        help="UPET model version.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to UPETCalculator, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local checkpoint path. If provided, avoids remote model resolution.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Start ASE row id (inclusive).",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="End ASE row id (inclusive). Default: all rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of structures to run.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N finished structures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output db if it already exists.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Optional JSONL file to store failed rows.",
    )
    args = parser.parse_args()

    if not args.input_db.exists():
        raise SystemExit(f"Input db not found: {args.input_db}")

    if args.output_db.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_db} already exists. Use --overwrite to replace it.")
        args.output_db.unlink()

    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    if args.error_log is None:
        args.error_log = args.output_db.with_suffix(".errors.jsonl")

    calc_kwargs = {
        "model": args.model,
        "version": args.version,
        "device": args.device,
    }
    if args.checkpoint_path is not None:
        if not args.checkpoint_path.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint_path}")
        calc_kwargs["checkpoint_path"] = str(args.checkpoint_path)

    calculator = UPETCalculator(**calc_kwargs)

    done = 0
    failed = 0
    with connect(args.input_db) as src, connect(args.output_db) as dst, args.error_log.open("w") as err_f:
        for row in src.select():
            if row.id < args.start_id:
                continue
            if args.end_id is not None and row.id > args.end_id:
                continue
            if args.limit is not None and done >= args.limit:
                break

            try:
                atoms = _clean_atoms(row)
                key_value_pairs, data = _extract_metadata(row)

                atoms.calc = calculator
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)

                calc_kwargs = {
                    "energy": energy,
                    "forces": forces,
                }
                try:
                    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float).reshape(-1)
                    if stress.size in (6, 9) and np.all(np.isfinite(stress)):
                        calc_kwargs["stress"] = stress
                except Exception:
                    pass

                atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
                dst.write(atoms, key_value_pairs=key_value_pairs, data=data)

                done += 1
                if args.log_every > 0 and done % args.log_every == 0:
                    print(f"[PET-MAD] finished {done} structures; last source row id = {row.id}", flush=True)

            except Exception as exc:
                failed += 1
                err_record = {
                    "source_db_row_id": int(row.id),
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"[PET-MAD] failed on source row id = {row.id}: {exc}", flush=True)

    print(
        json.dumps(
            {
                "input_db": str(args.input_db),
                "output_db": str(args.output_db),
                "model": args.model,
                "version": args.version,
                "device": args.device,
                "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else None,
                "finished": done,
                "failed": failed,
                "error_log": str(args.error_log),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
