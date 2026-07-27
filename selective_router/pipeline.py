from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_PRECISION_TARGETS, DEFAULT_TEACHER_PRIORITY, DEFAULT_TIERS
from .data import StructureRecord, load_labeled_records, load_unlabeled_records
from .features import build_pair_table, compute_structure_features, model_one_hot, compute_prediction_summary
from .modeling import (
    add_ood_features,
    apply_hierarchical_decision,
    assign_clusters,
    build_risk_coverage_curve,
    fit_binary_classifier,
    load_artifact_bundle,
    predict_probabilities,
    recall_at_k,
    save_artifact_bundle,
    split_groups,
    summarize_decisions,
    threshold_search,
)


TIER_ORDER = ["A", "B", "C"]


def _structure_feature_cols(pair_df: pd.DataFrame) -> list[str]:
    return [col for col in pair_df.columns if col.startswith("structure__") and "cluster_" not in col and col != "structure__cluster_id"]


def _candidate_feature_cols(pair_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in pair_df.columns
        if col.startswith("structure__")
        or col.startswith("model__")
    ]


def _router_feature_cols(pair_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in pair_df.columns
        if col.startswith("structure__")
        or col.startswith("model__")
        or col.startswith("router__")
    ]


def resolve_disagreement_models(mode: str | None, explicit_models: list[str] | None = None) -> list[str] | None:
    if explicit_models:
        return explicit_models
    if mode is None or mode == "all":
        return None
    if mode == "top3":
        return DEFAULT_TEACHER_PRIORITY[:3]
    if mode == "top5":
        return DEFAULT_TEACHER_PRIORITY[:5]
    raise ValueError(f"Unsupported disagreement mode: {mode}")


def _analysis_reports(pair_df: pd.DataFrame, output_dir: Path) -> None:
    cluster_frame = (
        pair_df.groupby(["structure__cluster_id", "model_name"], as_index=False)
        .agg(
            count=("structure_id", "nunique"),
            energy_mae_pa=("target__energy_err_pa", "mean"),
            energy_p90_pa=("target__energy_err_pa", lambda s: float(np.quantile(s.dropna(), 0.90))),
            force_mean_mae=("target__force_mean_err", "mean"),
            force_p90_mae=("target__force_p90_err", "mean"),
            tier_A_rate=("target__tier_A", "mean"),
            tier_B_rate=("target__tier_B", "mean"),
            tier_C_rate=("target__tier_C", "mean"),
        )
        .sort_values(["structure__cluster_id", "tier_B_rate"], ascending=[True, False])
    )
    cluster_frame.to_csv(output_dir / "cluster_model_profile.csv", index=False)


def train_experiment(
    root: Path,
    output_dir: Path,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    n_clusters: int = 12,
    max_records: int | None = None,
    backend: str = "random_forest",
    disagreement_mode: str = "all",
    disagreement_models: list[str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, model_names = load_labeled_records(root)
    if max_records is not None:
        records = records[:max_records]
    summary_models = resolve_disagreement_models(disagreement_mode, disagreement_models)
    pair_df = build_pair_table(
        records,
        model_names,
        DEFAULT_TIERS,
        disagreement_models=summary_models,
    )

    split = split_groups(pair_df, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    structure_cols = _structure_feature_cols(pair_df)
    pair_df, ood_artifact = add_ood_features(pair_df, split, structure_cols)
    pair_df, cluster_artifact = assign_clusters(pair_df, split, structure_cols, n_clusters=n_clusters, seed=seed)

    candidate_cols = _candidate_feature_cols(pair_df)
    router_cols = _router_feature_cols(pair_df)

    train_frame = pair_df[pair_df["structure_id"].isin(split.train_ids)].copy()
    val_frame = pair_df[pair_df["structure_id"].isin(split.val_ids)].copy()
    test_frame = pair_df[pair_df["structure_id"].isin(split.test_ids)].copy()

    candidate_model, candidate_info = fit_binary_classifier(
        train_frame, candidate_cols, "target__tier_B", seed=seed, backend=backend
    )
    for frame in (train_frame, val_frame, test_frame):
        frame["pred__candidate"] = predict_probabilities(candidate_model, frame, candidate_cols)

    tier_models = {}
    tier_infos = {}
    threshold_tables = {}
    chosen_thresholds = {}
    for tier in TIER_ORDER:
        target_col = f"target__tier_{tier}"
        model, info = fit_binary_classifier(
            train_frame, router_cols, target_col, seed=seed, backend=backend
        )
        tier_models[tier] = model
        tier_infos[tier] = info
        val_frame[f"pred__tier_{tier}"] = predict_probabilities(model, val_frame, router_cols)
        test_frame[f"pred__tier_{tier}"] = predict_probabilities(model, test_frame, router_cols)
        train_frame[f"pred__tier_{tier}"] = predict_probabilities(model, train_frame, router_cols)
        threshold, table = threshold_search(
            val_frame,
            prob_col=f"pred__tier_{tier}",
            label_col=target_col,
            min_precision=DEFAULT_PRECISION_TARGETS[tier],
        )
        chosen_thresholds[tier] = threshold
        threshold_tables[tier] = table
        table.to_csv(output_dir / f"threshold_search_tier_{tier}.csv", index=False)

    candidate_metrics = {
        "validation": recall_at_k(val_frame, "pred__candidate", "target__tier_B"),
        "test": recall_at_k(test_frame, "pred__candidate", "target__tier_B"),
    }
    decisions = apply_hierarchical_decision(test_frame, chosen_thresholds, TIER_ORDER)
    decision_summary = summarize_decisions(decisions)
    curve = build_risk_coverage_curve(test_frame, chosen_thresholds, TIER_ORDER)

    decisions.to_csv(output_dir / "test_decisions.csv", index=False)
    curve.to_csv(output_dir / "risk_coverage_curve.csv", index=False)
    _analysis_reports(pair_df, output_dir)

    artifact_bundle = {
        "seed": seed,
        "backend": backend,
        "model_names": model_names,
        "disagreement_mode": disagreement_mode,
        "disagreement_models": summary_models,
        "tiers": {name: asdict(tier) for name, tier in DEFAULT_TIERS.items()},
        "split": {
            "train_ids": sorted(split.train_ids),
            "val_ids": sorted(split.val_ids),
            "test_ids": sorted(split.test_ids),
        },
        "candidate_feature_cols": candidate_cols,
        "router_feature_cols": router_cols,
        "candidate_model": candidate_model,
        "router_models": tier_models,
        "thresholds": chosen_thresholds,
        "ood_artifact": ood_artifact,
        "cluster_artifact": cluster_artifact,
    }
    save_artifact_bundle(output_dir / "router_bundle.joblib", artifact_bundle)

    summary = {
        "backend": backend,
        "candidate_metrics": candidate_metrics,
        "decision_summary": decision_summary,
        "thresholds": chosen_thresholds,
        "disagreement_mode": disagreement_mode,
        "disagreement_models": summary_models,
        "train_structures": len(split.train_ids),
        "val_structures": len(split.val_ids),
        "test_structures": len(split.test_ids),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _single_record_pair_frame(
    record: StructureRecord,
    model_names: list[str],
    disagreement_models: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
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
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _augment_inference_features(pair_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    structure_cols = bundle["ood_artifact"]["structure_feature_cols"]
    structure_df = pair_df[["structure_id", *structure_cols]].drop_duplicates("structure_id").copy()

    all_x = structure_df[structure_cols].to_numpy(dtype=np.float64)
    ood_mean = bundle["ood_artifact"]["mean"]
    ood_scale = bundle["ood_artifact"]["scale"]
    ood_center = bundle["ood_artifact"]["center"]
    all_z = np.nan_to_num((all_x - ood_mean) / ood_scale, nan=0.0, posinf=0.0, neginf=0.0)
    structure_df["structure__ood_knn_mean"] = np.linalg.norm(all_z - ood_center, axis=1)

    cluster_mean = bundle["cluster_artifact"]["mean"]
    cluster_scale = bundle["cluster_artifact"]["scale"]
    centroids = bundle["cluster_artifact"]["centroids"]
    cluster_cols = bundle["cluster_artifact"]["cluster_cols"]
    cluster_z = np.nan_to_num((all_x - cluster_mean) / cluster_scale, nan=0.0, posinf=0.0, neginf=0.0)
    cluster_dist = np.sum((cluster_z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    cluster_ids = np.argmin(cluster_dist, axis=1)
    structure_df["structure__cluster_id"] = cluster_ids
    for cluster_id, col in enumerate(cluster_cols):
        structure_df[col] = (cluster_ids == cluster_id).astype(float)

    return pair_df.merge(
        structure_df[["structure_id", "structure__ood_knn_mean", "structure__cluster_id", *cluster_cols]],
        on="structure_id",
        how="left",
    )


def rank_models_for_structure(
    bundle_path: Path,
    structure_db: Path,
    structure_id: int,
) -> pd.DataFrame:
    bundle = load_artifact_bundle(bundle_path)
    records, _ = load_unlabeled_records(structure_db=structure_db, model_dbs=[])
    record_map = {record.structure_id: record for record in records}
    record = record_map[structure_id]
    pair_df = _single_record_pair_frame(
        record,
        bundle["model_names"],
        disagreement_models=bundle.get("disagreement_models"),
    )
    pair_df = _augment_inference_features(pair_df, bundle)
    pair_df["candidate_score"] = bundle["candidate_model"].predict_proba(
        pair_df[bundle["candidate_feature_cols"]].to_numpy()
    )[:, 1]
    cols = ["structure_id", "immutable_id", "model_name", "candidate_score"]
    return pair_df[cols].sort_values("candidate_score", ascending=False).reset_index(drop=True)


def route_structures(
    bundle_path: Path,
    structure_db: Path,
    model_dbs: list[Path],
    structure_ids: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    bundle = load_artifact_bundle(bundle_path)
    records, _ = load_unlabeled_records(structure_db=structure_db, model_dbs=model_dbs)
    if structure_ids is not None:
        record_set = set(structure_ids)
        records = [record for record in records if record.structure_id in record_set]

    frames = []
    for record in records:
        pair_df = _single_record_pair_frame(
            record,
            bundle["model_names"],
            disagreement_models=bundle.get("disagreement_models"),
        )
        frames.append(pair_df)
    all_pairs = pd.concat(frames, ignore_index=True)
    all_pairs = _augment_inference_features(all_pairs, bundle)

    for tier, model in bundle["router_models"].items():
        all_pairs[f"pred__tier_{tier}"] = model.predict_proba(
            all_pairs[bundle["router_feature_cols"]].to_numpy()
        )[:, 1]
    decision_thresholds = thresholds or bundle["thresholds"]
    decisions = apply_hierarchical_decision(all_pairs, decision_thresholds, TIER_ORDER)
    score_cols = ["structure_id", "immutable_id", "model_name", *[f"pred__tier_{tier}" for tier in TIER_ORDER]]
    scored_pairs = all_pairs[score_cols].copy()
    decisions = decisions.merge(
        scored_pairs[["structure_id", "model_name", "pred__tier_A", "pred__tier_B", "pred__tier_C"]],
        left_on=["structure_id", "chosen_model"],
        right_on=["structure_id", "model_name"],
        how="left",
    ).drop(columns=["model_name"])
    return decisions.sort_values("structure_id").reset_index(drop=True)
