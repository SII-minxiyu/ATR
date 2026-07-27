from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    ACTINIDES,
    ALKALI_METALS,
    ALKALINE_EARTHS,
    CHALCOGENS,
    HALOGENS,
    LANTHANIDES,
    METALLOIDS,
    NOBLE_GASES,
    PNICTOGENS,
    TRANSITION_METALS,
    TierSpec,
)
from .data import StructureRecord


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _cell_angles(cell: np.ndarray) -> tuple[float, float, float]:
    lengths = np.linalg.norm(cell, axis=1)
    angles = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        denom = lengths[i] * lengths[j]
        if denom <= 0:
            angles.append(float("nan"))
            continue
        cosine = np.dot(cell[i], cell[j]) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosine))))
    return angles[0], angles[1], angles[2]


def _minimum_image_distance_matrix(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    if len(positions) <= 1:
        return np.zeros((len(positions), len(positions)), dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    diff_frac = frac[:, None, :] - frac[None, :, :]
    diff_frac -= np.rint(diff_frac)
    diff_cart = diff_frac @ cell
    return np.linalg.norm(diff_cart, axis=-1)


def compute_structure_features(record: StructureRecord) -> dict[str, float]:
    numbers = record.numbers
    cell = record.cell
    positions = record.positions
    natoms = float(record.natoms)
    fractions = np.bincount(numbers, minlength=119).astype(np.float64)
    fractions /= natoms
    lengths = np.linalg.norm(cell, axis=1)
    alpha, beta, gamma = _cell_angles(cell)
    dist_matrix = _minimum_image_distance_matrix(positions, cell)
    iu = np.triu_indices(len(positions), k=1)
    pair_distances = dist_matrix[iu] if len(positions) > 1 else np.empty(0, dtype=np.float64)
    if len(positions) > 1:
        nn_matrix = dist_matrix.copy()
        np.fill_diagonal(nn_matrix, np.inf)
        nn_distances = np.min(nn_matrix, axis=1)
    else:
        nn_distances = np.empty(0, dtype=np.float64)

    feat = {
        "structure__natoms": natoms,
        "structure__unique_elements": float(np.count_nonzero(fractions)),
        "structure__volume": float(record.volume),
        "structure__volume_per_atom": float(record.volume / max(record.natoms, 1)),
        "structure__mass": float(record.mass),
        "structure__density": float(record.mass / record.volume) if record.volume > 0 else float("nan"),
        "structure__avg_z": float(np.mean(numbers)),
        "structure__std_z": float(np.std(numbers)),
        "structure__min_z": float(np.min(numbers)),
        "structure__max_z": float(np.max(numbers)),
        "structure__composition_entropy": float(
            -np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0]))
        ),
        "structure__frac_transition_metal": float(np.isin(numbers, list(TRANSITION_METALS)).mean()),
        "structure__frac_alkali": float(np.isin(numbers, list(ALKALI_METALS)).mean()),
        "structure__frac_alkaline_earth": float(np.isin(numbers, list(ALKALINE_EARTHS)).mean()),
        "structure__frac_halogen": float(np.isin(numbers, list(HALOGENS)).mean()),
        "structure__frac_chalcogen": float(np.isin(numbers, list(CHALCOGENS)).mean()),
        "structure__frac_pnictogen": float(np.isin(numbers, list(PNICTOGENS)).mean()),
        "structure__frac_lanthanide": float(np.isin(numbers, list(LANTHANIDES)).mean()),
        "structure__frac_actinide": float(np.isin(numbers, list(ACTINIDES)).mean()),
        "structure__frac_noble": float(np.isin(numbers, list(NOBLE_GASES)).mean()),
        "structure__frac_metalloid": float(np.isin(numbers, list(METALLOIDS)).mean()),
        "structure__cell_a": float(lengths[0]),
        "structure__cell_b": float(lengths[1]),
        "structure__cell_c": float(lengths[2]),
        "structure__cell_alpha": float(alpha),
        "structure__cell_beta": float(beta),
        "structure__cell_gamma": float(gamma),
        "structure__cell_min": float(np.min(lengths)),
        "structure__cell_max": float(np.max(lengths)),
        "structure__cell_mean": float(np.mean(lengths)),
        "structure__cell_ratio_max_min": float(np.max(lengths) / np.min(lengths)),
        "structure__pair_dist_p10": _safe_percentile(pair_distances, 0.10),
        "structure__pair_dist_p50": _safe_percentile(pair_distances, 0.50),
        "structure__pair_dist_p90": _safe_percentile(pair_distances, 0.90),
        "structure__pair_dist_mean": float(np.mean(pair_distances)) if pair_distances.size else float("nan"),
        "structure__nn_min": float(np.min(nn_distances)) if nn_distances.size else float("nan"),
        "structure__nn_mean": float(np.mean(nn_distances)) if nn_distances.size else float("nan"),
        "structure__nn_p50": _safe_percentile(nn_distances, 0.50),
        "structure__nn_p90": _safe_percentile(nn_distances, 0.90),
    }

    if pair_distances.size:
        for cutoff in (2.0, 2.5, 3.0):
            counts = np.sum(nn_matrix < cutoff, axis=1)
            feat[f"structure__coord_mean_{cutoff:.1f}"] = float(np.mean(counts))
            feat[f"structure__coord_p90_{cutoff:.1f}"] = float(np.quantile(counts, 0.90))
    else:
        for cutoff in (2.0, 2.5, 3.0):
            feat[f"structure__coord_mean_{cutoff:.1f}"] = float("nan")
            feat[f"structure__coord_p90_{cutoff:.1f}"] = float("nan")

    for z in range(1, 119):
        feat[f"structure__elem_frac_z{z:03d}"] = float(fractions[z])
    return feat


def _disagreement_model_names(
    record: StructureRecord,
    disagreement_models: Iterable[str] | None,
) -> list[str]:
    if disagreement_models is None:
        return list(record.predictions)
    allowed = set(disagreement_models)
    return [name for name in record.predictions if name in allowed]


def compute_prediction_summary(
    record: StructureRecord,
    target_model: str,
    disagreement_models: Iterable[str] | None = None,
) -> dict[str, float]:
    energy_values = []
    force_mag_means = []
    force_mag_p90s = []
    force_mag_maxs = []
    force_tensors = []
    valid_models = []
    summary_models = _disagreement_model_names(record, disagreement_models)

    for model_name, prediction in record.predictions.items():
        if model_name not in summary_models:
            continue
        energy = prediction.energy
        forces = prediction.forces
        energy_pa = float("nan")
        if energy is not None and np.isfinite(energy):
            energy_pa = float(energy / record.natoms)
            energy_values.append(energy_pa)
        if forces is not None:
            norms = np.linalg.norm(forces, axis=1)
            if np.isfinite(norms).all():
                force_mag_means.append(float(np.mean(norms)))
                force_mag_p90s.append(float(np.quantile(norms, 0.90)))
                force_mag_maxs.append(float(np.max(norms)))
                force_tensors.append(forces)
                valid_models.append(model_name)

    target_prediction = record.predictions.get(target_model)
    target_energy_pa = float("nan")
    target_force_mean = float("nan")
    target_force_p90 = float("nan")
    target_force_max = float("nan")
    target_valid_energy = 0.0
    target_valid_forces = 0.0
    if target_prediction is not None and target_prediction.energy is not None and np.isfinite(target_prediction.energy):
        target_valid_energy = 1.0
        target_energy_pa = float(target_prediction.energy / record.natoms)
    if target_prediction is not None and target_prediction.forces is not None:
        target_norms = np.linalg.norm(target_prediction.forces, axis=1)
        if np.isfinite(target_norms).all():
            target_valid_forces = 1.0
            target_force_mean = float(np.mean(target_norms))
            target_force_p90 = float(np.quantile(target_norms, 0.90))
            target_force_max = float(np.max(target_norms))

    energy_array = np.asarray(energy_values, dtype=np.float64)
    force_mean_array = np.asarray(force_mag_means, dtype=np.float64)
    pairwise_energy = []
    pairwise_force_mean = []
    pairwise_force_p90 = []
    pairwise_force_max = []

    valid_force_map = {
        name: pred.forces
        for name, pred in record.predictions.items()
        if name in summary_models
        if pred.forces is not None and np.isfinite(np.linalg.norm(pred.forces, axis=1)).all()
    }
    valid_force_names = sorted(valid_force_map)
    for i, left in enumerate(valid_force_names):
        for right in valid_force_names[i + 1 :]:
            diff = np.linalg.norm(valid_force_map[left] - valid_force_map[right], axis=1)
            pairwise_force_mean.append(float(np.mean(diff)))
            pairwise_force_p90.append(float(np.quantile(diff, 0.90)))
            pairwise_force_max.append(float(np.max(diff)))

    if energy_array.size >= 2:
        for i, value in enumerate(energy_array):
            for other in energy_array[i + 1 :]:
                pairwise_energy.append(abs(float(value - other)))

    energy_median = float(np.median(energy_array)) if energy_array.size else float("nan")
    target_vs_energy_median = abs(target_energy_pa - energy_median) if np.isfinite(target_energy_pa) else float("nan")

    force_ensemble_median = None
    if len(valid_force_map) >= 2:
        force_ensemble_median = np.median(
            np.stack([valid_force_map[name] for name in valid_force_names], axis=0), axis=0
        )
    target_force_disagreement = np.full(3, np.nan, dtype=np.float64)
    target_force_component_proxy = {
        "router__valid_force_model_count": float(len(valid_force_names)),
        "router__target_force_active_component_frac": float("nan"),
        "router__target_force_component_abs_ratio_mean": float("nan"),
        "router__target_force_component_abs_ratio_p10": float("nan"),
        "router__target_force_component_abs_ratio_p90": float("nan"),
        "router__target_force_component_abs_diff_mean": float("nan"),
        "router__target_force_component_abs_diff_p90": float("nan"),
        "router__target_force_component_sign_mismatch_ratio": float("nan"),
        "router__target_force_component_compress_count_proxy": float("nan"),
        "router__target_force_component_compress_ratio_proxy": float("nan"),
        "router__target_force_component_near_zero_ratio_proxy": float("nan"),
    }
    if force_ensemble_median is not None and target_prediction is not None and target_prediction.forces is not None:
        diff = np.linalg.norm(target_prediction.forces - force_ensemble_median, axis=1)
        if np.isfinite(diff).all():
            target_force_disagreement = np.array(
                [float(np.mean(diff)), float(np.quantile(diff, 0.90)), float(np.max(diff))],
                dtype=np.float64,
            )
        flat_target = target_prediction.forces.reshape(-1)
        flat_median = force_ensemble_median.reshape(-1)
        flat_abs_target = np.abs(flat_target)
        flat_abs_median = np.abs(flat_median)
        flat_abs_diff = np.abs(flat_target - flat_median)
        active_mask = flat_abs_median > 2.0
        if np.any(active_mask):
            abs_ratio = flat_abs_target[active_mask] / np.maximum(flat_abs_median[active_mask], 1e-8)
            target_force_component_proxy = {
                "router__valid_force_model_count": float(len(valid_force_names)),
                "router__target_force_active_component_frac": float(np.mean(active_mask)),
                "router__target_force_component_abs_ratio_mean": float(np.mean(abs_ratio)),
                "router__target_force_component_abs_ratio_p10": float(np.quantile(abs_ratio, 0.10)),
                "router__target_force_component_abs_ratio_p90": float(np.quantile(abs_ratio, 0.90)),
                "router__target_force_component_abs_diff_mean": float(np.mean(flat_abs_diff[active_mask])),
                "router__target_force_component_abs_diff_p90": float(np.quantile(flat_abs_diff[active_mask], 0.90)),
                "router__target_force_component_sign_mismatch_ratio": float(
                    np.mean(flat_target[active_mask] * flat_median[active_mask] < 0.0)
                ),
                "router__target_force_component_compress_count_proxy": float(
                    np.sum((flat_abs_target[active_mask] < 0.5 * flat_abs_median[active_mask]) & (flat_abs_diff[active_mask] > 2.0))
                ),
                "router__target_force_component_compress_ratio_proxy": float(
                    np.mean((flat_abs_target[active_mask] < 0.5 * flat_abs_median[active_mask]) & (flat_abs_diff[active_mask] > 2.0))
                ),
                "router__target_force_component_near_zero_ratio_proxy": float(
                    np.mean(flat_abs_target[active_mask] < 1.0)
                ),
            }

    feat = {
        "router__valid_model_count": float(len(set(valid_models) | {m for m in record.predictions if record.predictions[m].energy is not None})),
        "router__energy_ens_mean": float(np.mean(energy_array)) if energy_array.size else float("nan"),
        "router__energy_ens_std": float(np.std(energy_array)) if energy_array.size else float("nan"),
        "router__energy_ens_range": float(np.max(energy_array) - np.min(energy_array))
        if energy_array.size
        else float("nan"),
        "router__energy_ens_iqr": float(np.quantile(energy_array, 0.75) - np.quantile(energy_array, 0.25))
        if energy_array.size
        else float("nan"),
        "router__energy_pair_mean": float(np.mean(pairwise_energy)) if pairwise_energy else float("nan"),
        "router__energy_pair_p90": float(np.quantile(pairwise_energy, 0.90)) if pairwise_energy else float("nan"),
        "router__energy_pair_max": float(np.max(pairwise_energy)) if pairwise_energy else float("nan"),
        "router__force_mag_mean_mean": float(np.mean(force_mean_array)) if force_mean_array.size else float("nan"),
        "router__force_mag_mean_std": float(np.std(force_mean_array)) if force_mean_array.size else float("nan"),
        "router__force_pair_mean": float(np.mean(pairwise_force_mean)) if pairwise_force_mean else float("nan"),
        "router__force_pair_p90": float(np.quantile(pairwise_force_mean, 0.90)) if pairwise_force_mean else float("nan"),
        "router__force_pair_max": float(np.max(pairwise_force_max)) if pairwise_force_max else float("nan"),
        "router__target_energy_pa": target_energy_pa,
        "router__target_valid_energy": target_valid_energy,
        "router__target_force_mean": target_force_mean,
        "router__target_force_p90": target_force_p90,
        "router__target_force_max": target_force_max,
        "router__target_valid_forces": target_valid_forces,
        "router__target_energy_dev_to_median": target_vs_energy_median,
        "router__target_force_dev_mean": float(target_force_disagreement[0]),
        "router__target_force_dev_p90": float(target_force_disagreement[1]),
        "router__target_force_dev_max": float(target_force_disagreement[2]),
        **target_force_component_proxy,
    }
    return feat


def compute_pair_targets(
    record: StructureRecord,
    model_name: str,
    tiers: dict[str, TierSpec],
) -> dict[str, float]:
    prediction = record.predictions.get(model_name)
    if prediction is None:
        return {
            "target__valid_energy": 0.0,
            "target__valid_forces": 0.0,
            "target__energy_err_pa": float("nan"),
            "target__force_mean_err": float("nan"),
            "target__force_p90_err": float("nan"),
            "target__force_max_err": float("nan"),
            "target__tier_A": 0.0,
            "target__tier_B": 0.0,
            "target__tier_C": 0.0,
        }

    valid_energy = float(
        prediction.energy is not None and record.dft_energy is not None and np.isfinite(prediction.energy)
    )
    valid_forces = 0.0
    energy_err = float("nan")
    force_mean_err = float("nan")
    force_p90_err = float("nan")
    force_max_err = float("nan")
    if valid_energy:
        energy_err = abs(float(prediction.energy - record.dft_energy)) / record.natoms
    if prediction.forces is not None and record.dft_forces is not None:
        diff = np.linalg.norm(prediction.forces - record.dft_forces, axis=1)
        if np.isfinite(diff).all():
            valid_forces = 1.0
            force_mean_err = float(np.mean(diff))
            force_p90_err = float(np.quantile(diff, 0.90))
            force_max_err = float(np.max(diff))

    labels = {}
    for tier_name, tier in tiers.items():
        ok = valid_energy == 1.0 and energy_err <= tier.energy_pa_max
        if tier.force_mean_max is not None:
            ok = ok and valid_forces == 1.0 and force_mean_err <= tier.force_mean_max
        if tier.force_p90_max is not None:
            ok = ok and valid_forces == 1.0 and force_p90_err <= tier.force_p90_max
        labels[f"target__tier_{tier_name}"] = float(ok)

    return {
        "target__valid_energy": valid_energy,
        "target__valid_forces": valid_forces,
        "target__energy_err_pa": energy_err,
        "target__force_mean_err": force_mean_err,
        "target__force_p90_err": force_p90_err,
        "target__force_max_err": force_max_err,
        **labels,
    }


def model_one_hot(model_name: str, model_names: Iterable[str]) -> dict[str, float]:
    return {f"model__{name}": float(name == model_name) for name in model_names}


def build_pair_table(
    records: list[StructureRecord],
    model_names: list[str],
    tiers: dict[str, TierSpec],
    disagreement_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for record in records:
        structure_feat = compute_structure_features(record)
        for model_name in model_names:
            row = {
                "structure_id": record.structure_id,
                "immutable_id": record.immutable_id,
                "group_id": record.group_id,
                "model_name": model_name,
                **structure_feat,
                **model_one_hot(model_name, model_names),
                **compute_prediction_summary(record, model_name, disagreement_models=disagreement_models),
                **compute_pair_targets(record, model_name, tiers),
            }
            rows.append(row)
    return pd.DataFrame(rows)
