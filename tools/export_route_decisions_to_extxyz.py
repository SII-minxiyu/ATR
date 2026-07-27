from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ase.db import connect
from ase.io import write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export A/R routing decisions back to an ExtXYZ file with route labels."
    )
    parser.add_argument("--structure-db", type=Path, required=True, help="ASE db providing the structures.")
    parser.add_argument("--decisions-csv", type=Path, required=True, help="CSV from `ar-route`.")
    parser.add_argument("--output-xyz", type=Path, required=True, help="Output .xyz/.extxyz path.")
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Only export accepted structures. Default exports all structures with route labels.",
    )
    args = parser.parse_args()

    decisions = pd.read_csv(args.decisions_csv)
    if "structure_id" not in decisions.columns:
        raise SystemExit("decisions csv must contain a `structure_id` column")

    if args.accepted_only:
        decisions = decisions[decisions["status"] == "accept"].copy()

    decision_cols = [
        "status",
        "chosen_model",
        "chosen_label",
        "confidence",
        "postprocess_reject",
        "postprocess_reason",
    ]
    keep_cols = ["structure_id"] + [c for c in decision_cols if c in decisions.columns]
    decisions = decisions[keep_cols].copy()
    decisions = decisions.drop_duplicates(subset=["structure_id"])
    decision_map = decisions.set_index("structure_id").to_dict(orient="index")

    args.output_xyz.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with connect(args.structure_db) as db:
        with open(args.output_xyz, "w", encoding="utf-8") as handle:
            for row in db.select():
                info = decision_map.get(int(row.id))
                if info is None:
                    continue
                atoms = row.toatoms()
                atoms.info["route_status"] = str(info.get("status", ""))
                chosen_model = info.get("chosen_model")
                atoms.info["route_chosen_model"] = "" if pd.isna(chosen_model) else str(chosen_model)
                chosen_label = info.get("chosen_label")
                atoms.info["route_chosen_label"] = "" if pd.isna(chosen_label) else str(chosen_label)
                confidence = info.get("confidence")
                if confidence is not None and not pd.isna(confidence):
                    atoms.info["route_confidence"] = float(confidence)
                post_reject = info.get("postprocess_reject")
                if post_reject is not None and not pd.isna(post_reject):
                    atoms.info["route_postprocess_reject"] = bool(post_reject)
                post_reason = info.get("postprocess_reason")
                if post_reason is not None and not pd.isna(post_reason):
                    atoms.info["route_postprocess_reason"] = str(post_reason)
                write(handle, atoms, format="extxyz")
                written += 1

    print(f"Wrote {written} structures to {args.output_xyz}")


if __name__ == "__main__":
    main()
