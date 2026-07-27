from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write

from selective_router.ar_maxforce import (
    LIGHT_POSTPROCESS_CONFIDENCE_LT,
    LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
    _select_best_model,
    apply_light_postprocess,
)
from selective_router.data import ModelPrediction, StructureRecord
from selective_router.modeling import load_artifact_bundle
from selective_router.pipeline import _augment_inference_features, _single_record_pair_frame


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


def _structure_id_from_atoms(atoms, fallback: int) -> tuple[int, str]:
    immutable_id = (
        atoms.info.get("immutable_id")
        or atoms.info.get("IMMUTABLE_ID")
        or atoms.info.get("id")
        or atoms.info.get("ID")
        or f"xyz-{fallback}"
    )
    return fallback, str(immutable_id)


def _label_from_atoms(atoms) -> tuple[str | None, str | None]:
    sid = atoms.info.get("id") or atoms.info.get("ID")
    immutable = atoms.info.get("immutable_id") or atoms.info.get("IMMUTABLE_ID")
    sid = None if sid is None else str(sid)
    immutable = None if immutable is None else str(immutable)
    return sid, immutable


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


def _build_record(structure_idx: int, anchor_atoms, teacher_atoms: dict[str, object]) -> StructureRecord:
    sid, immutable_id = _structure_id_from_atoms(anchor_atoms, structure_idx)
    predictions = {}
    for model_name, atoms in teacher_atoms.items():
        predictions[model_name] = ModelPrediction(
            name=model_name,
            energy=_extract_energy(atoms),
            forces=_extract_forces(atoms),
        )
    return StructureRecord(
        structure_id=sid,
        immutable_id=immutable_id,
        group_id=immutable_id,
        natoms=len(anchor_atoms),
        mass=float(np.sum(anchor_atoms.get_masses())),
        volume=float(abs(np.linalg.det(anchor_atoms.cell.array))),
        numbers=np.asarray(anchor_atoms.numbers, dtype=np.int32),
        positions=np.asarray(anchor_atoms.positions, dtype=float),
        cell=np.asarray(anchor_atoms.cell.array, dtype=float),
        dft_energy=None,
        dft_forces=None,
        predictions=predictions,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly route 5 teacher extxyz files without converting to ASE db first."
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Path to ar_router_seed2026.joblib")
    parser.add_argument(
        "--teacher",
        action="append",
        type=_parse_teacher_arg,
        required=True,
        help="Repeat as MODEL_NAME=/path/to/teacher.extxyz",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Where to write routing decisions CSV.")
    parser.add_argument(
        "--output-accepted-xyz",
        type=Path,
        default=None,
        help="Optional accepted-only extxyz with chosen teacher energy/forces attached.",
    )
    parser.add_argument(
        "--output-all-labeled-xyz",
        type=Path,
        default=None,
        help="Optional all-structures extxyz with route labels attached.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--postprocess-light",
        action="store_true",
        help="Apply the lightweight teacher-only postprocess reject rule after routing.",
    )
    parser.add_argument(
        "--postprocess-confidence-lt",
        type=float,
        default=LIGHT_POSTPROCESS_CONFIDENCE_LT,
    )
    parser.add_argument(
        "--postprocess-force-dev-max-gt",
        type=float,
        default=LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.bundle)
    expected = list(bundle["model_names"])
    provided = [name for name, _ in args.teacher]
    if set(provided) != set(expected):
        raise SystemExit(
            f"Teacher set mismatch. Bundle expects {expected}, but got {provided}."
        )

    ordered_specs = [(name, dict(args.teacher)[name]) for name in expected]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_accepted_xyz is not None:
        args.output_accepted_xyz.parent.mkdir(parents=True, exist_ok=True)
    if args.output_all_labeled_xyz is not None:
        args.output_all_labeled_xyz.parent.mkdir(parents=True, exist_ok=True)

    accepted_handle = None
    all_handle = None
    if args.output_accepted_xyz is not None:
        accepted_handle = open(args.output_accepted_xyz, "w", encoding="utf-8")
    if args.output_all_labeled_xyz is not None:
        all_handle = open(args.output_all_labeled_xyz, "w", encoding="utf-8")

    decisions_rows: list[dict[str, object]] = []
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])

    try:
        for idx, anchor_atoms, teacher_atoms in _iter_aligned_teacher_atoms(ordered_specs):
            record = _build_record(idx, anchor_atoms, teacher_atoms)
            pair_df = _single_record_pair_frame(
                record,
                bundle["model_names"],
                disagreement_models=bundle.get("disagreement_models"),
            )
            pair_df = _augment_inference_features(pair_df, bundle)
            pair_df["pred__tier_A"] = bundle["router_model"].predict_proba(
                pair_df[bundle["router_feature_cols"]].to_numpy()
            )[:, 1]

            decision = _select_best_model(pair_df, prob_col="pred__tier_A", threshold=threshold)
            decision["threshold"] = threshold

            ranked_pairs = (
                pair_df[["structure_id", "immutable_id", "model_name", "pred__tier_A"]]
                .copy()
                .sort_values(["structure_id", "pred__tier_A"], ascending=[True, False])
            )
            top3 = ranked_pairs.groupby("structure_id").head(3).copy()
            top3["rank"] = top3.groupby("structure_id").cumcount() + 1
            pivot = top3.pivot(index="structure_id", columns="rank", values=["model_name", "pred__tier_A"])
            pivot.columns = [
                f"top{rank}_{'model' if field == 'model_name' else 'score'}"
                for field, rank in pivot.columns
            ]
            pivot = pivot.reset_index()
            decision = decision.merge(pivot, on="structure_id", how="left")

            if args.postprocess_light:
                decision = apply_light_postprocess(
                    decision,
                    pair_df,
                    confidence_lt=args.postprocess_confidence_lt,
                    force_dev_max_gt=args.postprocess_force_dev_max_gt,
                )

            row = decision.iloc[0].to_dict()
            decisions_rows.append(row)

            route_labels = {
                "route_status": str(row.get("status", "")),
                "route_chosen_model": "" if pd.isna(row.get("chosen_model")) else str(row.get("chosen_model")),
                "route_chosen_label": str(row.get("chosen_label", "")),
                "route_confidence": float(row.get("confidence", float("nan"))),
                "route_postprocess_reject": bool(row.get("postprocess_reject", False)),
                "route_postprocess_reason": "" if pd.isna(row.get("postprocess_reason")) else str(row.get("postprocess_reason")),
            }

            if all_handle is not None:
                atoms_all = anchor_atoms.copy()
                atoms_all.info.update(route_labels)
                write(all_handle, atoms_all, format="extxyz")

            if accepted_handle is not None and row.get("status") == "accept":
                chosen_model = str(row["chosen_model"])
                source_atoms = teacher_atoms[chosen_model]
                atoms_acc = source_atoms.copy()
                _attach_prediction_calc(atoms_acc, source_atoms=source_atoms)
                atoms_acc.info.update(route_labels)
                write(accepted_handle, atoms_acc, format="extxyz")

            if args.log_every > 0 and idx % args.log_every == 0:
                print(f"[route-extxyz-top5] finished {idx} structures")
            if args.limit is not None and idx >= args.limit:
                break
    finally:
        if accepted_handle is not None:
            accepted_handle.close()
        if all_handle is not None:
            all_handle.close()

    decisions = pd.DataFrame(decisions_rows).sort_values("structure_id").reset_index(drop=True)
    decisions.to_csv(args.output_csv, index=False)
    summary = {
        "bundle": str(args.bundle),
        "teacher_files": {name: str(path) for name, path in ordered_specs},
        "num_structures": int(len(decisions)),
        "accept_count": int((decisions["status"] == "accept").sum()) if len(decisions) else 0,
        "reject_count": int((decisions["status"] == "reject").sum()) if len(decisions) else 0,
        "threshold": threshold,
        "postprocess_light": bool(args.postprocess_light),
        "output_csv": str(args.output_csv),
        "output_accepted_xyz": str(args.output_accepted_xyz) if args.output_accepted_xyz else None,
        "output_all_labeled_xyz": str(args.output_all_labeled_xyz) if args.output_all_labeled_xyz else None,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
