from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import StructureRecord, load_labeled_records
from .features import compute_structure_features, model_one_hot
from .pipeline import _augment_inference_features


ELEMENT_SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
SYMBOL_TO_Z = {symbol: idx for idx, symbol in enumerate(ELEMENT_SYMBOLS) if symbol}

LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
IMMUTABLE_RE = re.compile(r"immutable_id=([^\s]+)")
ID_RE = re.compile(r"\bid=([^\s]+)")
PROPERTIES_RE = re.compile(r"Properties=([^\s]+)")


def _mass_lookup(root: Path) -> dict[int, float]:
    bundled = Path(__file__).with_name("atomic_mass_lookup.json")
    if bundled.exists():
        data = json.loads(bundled.read_text())
        return {int(k): float(v) for k, v in data.items()}

    records, _ = load_labeled_records(root)
    elements = sorted({int(z) for record in records for z in np.unique(record.numbers)})
    z_to_idx = {z: idx for idx, z in enumerate(elements)}
    a = np.zeros((len(records), len(elements)), dtype=np.float64)
    b = np.zeros(len(records), dtype=np.float64)
    for row_idx, record in enumerate(records):
        numbers, counts = np.unique(record.numbers, return_counts=True)
        for z, count in zip(numbers, counts):
            a[row_idx, z_to_idx[int(z)]] = float(count)
        b[row_idx] = float(record.mass)
    masses, *_ = np.linalg.lstsq(a, b, rcond=None)
    return {z: float(masses[idx]) for z, idx in z_to_idx.items()}


def _parse_properties(comment_line: str) -> list[tuple[str, int]]:
    match = PROPERTIES_RE.search(comment_line)
    if match is None:
        return [("species", 1), ("pos", 3)]
    parts = match.group(1).split(":")
    fields = []
    for idx in range(0, len(parts), 3):
        name = parts[idx]
        width = int(parts[idx + 2])
        fields.append((name, width))
    return fields


def _comment_metadata(comment_line: str) -> tuple[np.ndarray, str, str]:
    lattice_match = LATTICE_RE.search(comment_line)
    if lattice_match is None:
        raise ValueError("ExtXYZ comment line missing Lattice field.")
    lattice = np.fromstring(lattice_match.group(1), sep=" ", dtype=np.float64).reshape(3, 3)
    immutable = ""
    m = IMMUTABLE_RE.search(comment_line)
    if m is not None:
        immutable = m.group(1)
    structure_id = ""
    m = ID_RE.search(comment_line)
    if m is not None:
        structure_id = m.group(1)
    return lattice, immutable, structure_id


def iter_extxyz_records(
    xyz_path: Path,
    root: Path,
    *,
    limit: int | None = None,
    every_n: int = 1,
    offset: int = 0,
) -> StructureRecord:
    masses = _mass_lookup(root)
    with xyz_path.open("r") as handle:
        structure_index = 0
        yielded = 0
        while True:
            first = handle.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            natoms = int(first)
            comment = handle.readline().strip()
            cell, immutable_id, raw_id = _comment_metadata(comment)
            fields = _parse_properties(comment)

            symbols = []
            positions = np.zeros((natoms, 3), dtype=np.float64)
            for atom_idx in range(natoms):
                tokens = handle.readline().split()
                cursor = 0
                symbol = None
                pos = None
                for name, width in fields:
                    values = tokens[cursor : cursor + width]
                    cursor += width
                    if name == "species":
                        symbol = values[0]
                    elif name == "pos":
                        pos = [float(v) for v in values]
                if symbol is None or pos is None:
                    raise ValueError("ExtXYZ parser could not find species/pos fields.")
                symbols.append(symbol)
                positions[atom_idx] = pos

            numbers = np.asarray([SYMBOL_TO_Z[symbol] for symbol in symbols], dtype=np.int32)
            structure_index += 1
            immutable = immutable_id or raw_id or f"xyz-{structure_index}"
            volume = float(abs(np.linalg.det(cell)))
            mass = float(sum(masses.get(int(z), float(z)) for z in numbers))
            if every_n <= 1 or ((structure_index - 1 - offset) % every_n == 0):
                yield StructureRecord(
                    structure_id=structure_index,
                    immutable_id=immutable,
                    group_id=immutable,
                    natoms=natoms,
                    mass=mass,
                    volume=volume,
                    numbers=numbers,
                    positions=positions,
                    cell=cell,
                    dft_energy=None,
                    dft_forces=None,
                    predictions={},
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    break


def rank_extxyz_candidates(
    bundle_path: Path,
    xyz_path: Path,
    root: Path,
    output_dir: Path,
    *,
    top_k: int = 3,
    limit: int | None = None,
    every_n: int = 1,
    offset: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(bundle_path)
    model_names = bundle["model_names"]
    top_k = min(top_k, len(model_names))

    top_rows = []
    top1_counter: Counter[str] = Counter()
    top1_scores: list[float] = []
    processed = 0

    for record in iter_extxyz_records(
        xyz_path,
        root=root,
        limit=limit,
        every_n=every_n,
        offset=offset,
    ):
        structure_feat = compute_structure_features(record)
        rows = []
        for model_name in model_names:
            rows.append(
                {
                    "structure_id": record.structure_id,
                    "immutable_id": record.immutable_id,
                    "group_id": record.group_id,
                    "model_name": model_name,
                    **structure_feat,
                    **model_one_hot(model_name, model_names),
                }
            )
        pair_df = pd.DataFrame(rows)
        pair_df = _augment_inference_features(pair_df, bundle)
        scores = bundle["candidate_model"].predict_proba(
            pair_df[bundle["candidate_feature_cols"]].to_numpy()
        )[:, 1]
        pair_df["candidate_score"] = scores
        ranked = pair_df.sort_values("candidate_score", ascending=False).reset_index(drop=True)
        top1 = ranked.iloc[0]
        top1_counter[str(top1["model_name"])] += 1
        top1_scores.append(float(top1["candidate_score"]))

        row = {
            "structure_id": int(record.structure_id),
            "immutable_id": record.immutable_id,
            "natoms": int(record.natoms),
            "ood": float(ranked.iloc[0]["structure__ood_knn_mean"]),
        }
        for rank in range(top_k):
            entry = ranked.iloc[rank]
            row[f"top{rank + 1}_model"] = str(entry["model_name"])
            row[f"top{rank + 1}_score"] = float(entry["candidate_score"])
        top_rows.append(row)
        processed += 1

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(output_dir / "candidate_topk.csv", index=False)

    summary = {
        "xyz_path": str(xyz_path),
        "bundle_path": str(bundle_path),
        "processed_structures": int(processed),
        "top_k": int(top_k),
        "every_n": int(every_n),
        "offset": int(offset),
        "top1_model_counts": dict(top1_counter),
        "top1_score_mean": float(np.mean(top1_scores)) if top1_scores else float("nan"),
        "top1_score_p10": float(np.quantile(top1_scores, 0.10)) if top1_scores else float("nan"),
        "top1_score_p50": float(np.quantile(top1_scores, 0.50)) if top1_scores else float("nan"),
        "top1_score_p90": float(np.quantile(top1_scores, 0.90)) if top1_scores else float("nan"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
