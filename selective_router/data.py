from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import DEFAULT_MODEL_FILES, REFERENCE_DB, normalize_model_name


@dataclass
class ModelPrediction:
    name: str
    energy: float | None
    forces: np.ndarray | None


@dataclass
class StructureRecord:
    structure_id: int
    immutable_id: str
    group_id: str
    natoms: int
    mass: float
    volume: float
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    dft_energy: float | None
    dft_forces: np.ndarray | None
    predictions: dict[str, ModelPrediction]


def _decode_int_array(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.int32).copy()


def _decode_float_matrix(blob: bytes | None, width: int) -> np.ndarray | None:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float64).copy()
    return arr.reshape(-1, width)


def _decode_float_vector(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float64).copy()


def _load_structure_rows(
    db_path: Path,
    *,
    include_geometry: bool,
    include_predictions: bool,
    include_immutable: bool,
) -> dict[int, dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    immutable_expr = "null"
    if include_immutable:
        has_text = cur.execute(
            "select count(*) from sqlite_master where type='table' and name='text_key_values'"
        ).fetchone()[0]
        immutable_expr = (
            "(select value from text_key_values where id = s.id and key = 'IMMUTABLE_ID' limit 1)"
            if has_text
            else "null"
        )
    geometry_numbers = "s.numbers" if include_geometry else "null"
    geometry_positions = "s.positions" if include_geometry else "null"
    geometry_cell = "s.cell" if include_geometry else "null"
    prediction_energy = "s.energy" if include_predictions else "null"
    prediction_forces = "s.forces" if include_predictions else "null"
    query = f"""
        select
            s.id,
            s.natoms,
            s.mass,
            s.volume,
            {geometry_numbers} as numbers,
            {geometry_positions} as positions,
            {geometry_cell} as cell,
            {prediction_energy} as energy,
            {prediction_forces} as forces,
            {immutable_expr} as immutable_id
        from systems s
        order by s.id
    """
    rows = {}
    for row in cur.execute(query):
        rows[row["id"]] = {
            "structure_id": int(row["id"]),
            "natoms": int(row["natoms"]),
            "mass": float(row["mass"]),
            "volume": float(row["volume"]),
            "numbers": _decode_int_array(row["numbers"]),
            "positions": _decode_float_matrix(row["positions"], 3),
            "cell": _decode_float_matrix(row["cell"], 3),
            "energy": float(row["energy"]) if row["energy"] is not None else None,
            "forces": _decode_float_matrix(row["forces"], 3),
            "immutable_id": row["immutable_id"] or "",
        }
    conn.close()
    return rows


def _structure_hash(numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(np.asarray(numbers, dtype=np.int32).tobytes())
    digest.update(np.asarray(positions, dtype=np.float64).tobytes())
    digest.update(np.asarray(cell, dtype=np.float64).tobytes())
    return digest.hexdigest()


def resolve_default_paths(root: Path) -> tuple[Path, list[Path]]:
    reference = root / REFERENCE_DB
    model_paths = [root / filename for filename in DEFAULT_MODEL_FILES]
    return reference, model_paths


def load_labeled_records(
    root: Path,
    reference_db: Path | None = None,
    model_dbs: Iterable[Path] | None = None,
    verify_alignment: bool = False,
) -> tuple[list[StructureRecord], list[str]]:
    reference_path, default_models = resolve_default_paths(root)
    reference_db = reference_db or reference_path
    model_paths = list(model_dbs or default_models)

    reference_rows = _load_structure_rows(
        reference_db,
        include_geometry=True,
        include_predictions=True,
        include_immutable=False,
    )
    anchor_rows = _load_structure_rows(
        model_paths[0],
        include_geometry=verify_alignment,
        include_predictions=False,
        include_immutable=True,
    )

    model_rows_by_name = {}
    model_names = []
    for model_path in model_paths:
        model_name = normalize_model_name(model_path)
        model_names.append(model_name)
        model_rows_by_name[model_name] = _load_structure_rows(
            model_path,
            include_geometry=verify_alignment,
            include_predictions=True,
            include_immutable=False,
        )

    records: list[StructureRecord] = []
    for structure_id, ref_row in reference_rows.items():
        anchor = anchor_rows[structure_id]
        if verify_alignment:
            if not np.array_equal(ref_row["numbers"], anchor["numbers"]):
                raise ValueError(f"numbers mismatch for id={structure_id}")
            if not np.array_equal(ref_row["positions"], anchor["positions"]):
                raise ValueError(f"positions mismatch for id={structure_id}")
            if not np.array_equal(ref_row["cell"], anchor["cell"]):
                raise ValueError(f"cell mismatch for id={structure_id}")

        predictions = {}
        for model_name, model_rows in model_rows_by_name.items():
            model_row = model_rows[structure_id]
            if verify_alignment:
                if not np.array_equal(anchor["numbers"], model_row["numbers"]):
                    raise ValueError(f"{model_name}: numbers mismatch for id={structure_id}")
                if not np.array_equal(anchor["positions"], model_row["positions"]):
                    raise ValueError(f"{model_name}: positions mismatch for id={structure_id}")
                if not np.array_equal(anchor["cell"], model_row["cell"]):
                    raise ValueError(f"{model_name}: cell mismatch for id={structure_id}")
            predictions[model_name] = ModelPrediction(
                name=model_name,
                energy=model_row["energy"],
                forces=model_row["forces"],
            )

        immutable_id = anchor["immutable_id"] or f"id-{structure_id}"
        group_id = immutable_id or _structure_hash(ref_row["numbers"], ref_row["positions"], ref_row["cell"])
        records.append(
            StructureRecord(
                structure_id=structure_id,
                immutable_id=immutable_id,
                group_id=group_id,
                natoms=ref_row["natoms"],
                mass=ref_row["mass"],
                volume=ref_row["volume"],
                numbers=ref_row["numbers"],
                positions=ref_row["positions"],
                cell=ref_row["cell"],
                dft_energy=ref_row["energy"],
                dft_forces=ref_row["forces"],
                predictions=predictions,
            )
        )
    return records, model_names


def load_unlabeled_records(
    structure_db: Path,
    model_dbs: Iterable[Path] | None = None,
    verify_alignment: bool = False,
) -> tuple[list[StructureRecord], list[str]]:
    source_rows = _load_structure_rows(
        structure_db,
        include_geometry=True,
        include_predictions=False,
        include_immutable=True,
    )
    model_paths = list(model_dbs or [])
    model_rows_by_name = {}
    model_names = []
    for model_path in model_paths:
        model_name = normalize_model_name(model_path)
        model_names.append(model_name)
        model_rows_by_name[model_name] = _load_structure_rows(
            model_path,
            include_geometry=verify_alignment,
            include_predictions=True,
            include_immutable=False,
        )

    records: list[StructureRecord] = []
    for structure_id, source_row in source_rows.items():
        predictions = {}
        for model_name, model_rows in model_rows_by_name.items():
            if structure_id not in model_rows:
                continue
            model_row = model_rows[structure_id]
            if verify_alignment:
                if not np.array_equal(source_row["numbers"], model_row["numbers"]):
                    raise ValueError(f"{model_name}: numbers mismatch for id={structure_id}")
                if not np.array_equal(source_row["positions"], model_row["positions"]):
                    raise ValueError(f"{model_name}: positions mismatch for id={structure_id}")
                if not np.array_equal(source_row["cell"], model_row["cell"]):
                    raise ValueError(f"{model_name}: cell mismatch for id={structure_id}")
            predictions[model_name] = ModelPrediction(
                name=model_name,
                energy=model_row["energy"],
                forces=model_row["forces"],
            )

        immutable_id = source_row["immutable_id"] or f"id-{structure_id}"
        group_id = immutable_id or _structure_hash(
            source_row["numbers"], source_row["positions"], source_row["cell"]
        )
        records.append(
            StructureRecord(
                structure_id=structure_id,
                immutable_id=immutable_id,
                group_id=group_id,
                natoms=source_row["natoms"],
                mass=source_row["mass"],
                volume=source_row["volume"],
                numbers=source_row["numbers"],
                positions=source_row["positions"],
                cell=source_row["cell"],
                dft_energy=None,
                dft_forces=None,
                predictions=predictions,
            )
        )
    return records, model_names
