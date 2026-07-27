from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class SplitBundle:
    train_ids: set[int]
    val_ids: set[int]
    test_ids: set[int]


def split_groups(
    pair_df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> SplitBundle:
    groups = pair_df[["structure_id", "group_id"]].drop_duplicates()
    unique_groups = groups["group_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(round(train_fraction * n_groups))
    n_val = int(round(val_fraction * n_groups))
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    def ids_for(group_set: set[str]) -> set[int]:
        subset = groups[groups["group_id"].isin(group_set)]
        return set(subset["structure_id"].astype(int).tolist())

    return SplitBundle(
        train_ids=ids_for(train_groups),
        val_ids=ids_for(val_groups),
        test_ids=ids_for(test_groups),
    )


def add_ood_features(
    pair_df: pd.DataFrame,
    split: SplitBundle,
    structure_feature_cols: list[str],
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    structure_df = pair_df[["structure_id", *structure_feature_cols]].drop_duplicates("structure_id").copy()
    train_mask = structure_df["structure_id"].isin(split.train_ids)
    train_struct = structure_df.loc[train_mask].copy()

    train_x = train_struct[structure_feature_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where((np.isfinite(scale)) & (scale > 1e-12), scale, 1.0)
    train_z = np.nan_to_num((train_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    center = np.mean(train_z, axis=0)

    all_x = structure_df[structure_feature_cols].to_numpy(dtype=np.float64)
    all_z = np.nan_to_num((all_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    structure_df["structure__ood_knn_mean"] = np.linalg.norm(all_z - center, axis=1)
    pair_df = pair_df.merge(
        structure_df[["structure_id", "structure__ood_knn_mean"]],
        on="structure_id",
        how="left",
    )
    artifact = {
        "structure_feature_cols": structure_feature_cols,
        "mean": mean,
        "scale": scale,
        "center": center,
    }
    return pair_df, artifact


def assign_clusters(
    pair_df: pd.DataFrame,
    split: SplitBundle,
    structure_feature_cols: list[str],
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    structure_df = pair_df[["structure_id", *structure_feature_cols]].drop_duplicates("structure_id").copy()
    train_mask = structure_df["structure_id"].isin(split.train_ids)
    train_x_raw = structure_df.loc[train_mask, structure_feature_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(train_x_raw, axis=0)
    scale = np.nanstd(train_x_raw, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where((np.isfinite(scale)) & (scale > 1e-12), scale, 1.0)
    train_x = np.nan_to_num((train_x_raw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    n_clusters = min(n_clusters, max(2, len(train_x) // 50))
    rng = np.random.default_rng(seed)
    initial_ids = rng.choice(len(train_x), size=n_clusters, replace=False)
    centroids = train_x[initial_ids].copy()
    for _ in range(12):
        distances = np.sum((train_x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            members = train_x[labels == cluster_id]
            if len(members):
                new_centroids[cluster_id] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    all_x = structure_df[structure_feature_cols].to_numpy(dtype=np.float64)
    all_z = np.nan_to_num((all_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    all_distances = np.sum((all_z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    cluster_ids = np.argmin(all_distances, axis=1)
    structure_df["structure__cluster_id"] = cluster_ids
    for cluster_id in range(n_clusters):
        structure_df[f"structure__cluster_{cluster_id:02d}"] = (
            structure_df["structure__cluster_id"] == cluster_id
        ).astype(float)
    cluster_cols = [f"structure__cluster_{cluster_id:02d}" for cluster_id in range(n_clusters)]
    pair_df = pair_df.merge(
        structure_df[["structure_id", "structure__cluster_id", *cluster_cols]],
        on="structure_id",
        how="left",
    )
    artifact = {
        "feature_cols": structure_feature_cols,
        "mean": mean,
        "scale": scale,
        "centroids": centroids,
        "cluster_cols": cluster_cols,
    }
    return pair_df, artifact


def fit_binary_classifier(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
    backend: str = "random_forest",
) -> tuple[Pipeline, dict]:
    x_train = frame[feature_cols].to_numpy()
    y_train = frame[target_col].to_numpy(dtype=int)
    positive = float(np.mean(y_train))
    backend = backend.lower()
    if backend in {"random_forest", "rf"}:
        estimator: Any = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    elif backend in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("XGBoost backend requested but xgboost is not importable.") from exc
        negatives = max(1, len(y_train) - int(y_train.sum()))
        positives = max(1, int(y_train.sum()))
        estimator = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=4.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            scale_pos_weight=float(negatives / positives),
        )
    elif backend in {"lightgbm", "lgbm", "lgb"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("LightGBM backend requested but lightgbm is not importable.") from exc
        estimator = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            objective="binary",
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )
    model.fit(x_train, y_train)
    info = {"positive_rate": positive, "backend": backend}
    return model, info


def predict_probabilities(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    return model.predict_proba(frame[feature_cols].to_numpy())[:, 1]


def threshold_search(
    frame: pd.DataFrame,
    prob_col: str,
    label_col: str,
    min_precision: float,
) -> tuple[float, pd.DataFrame]:
    results = []
    thresholds = np.linspace(0.05, 0.99, 95)
    structure_ids = frame["structure_id"].unique()
    tier = label_col.rsplit("_", 1)[-1]
    for threshold in thresholds:
        accepted = 0
        hits = 0
        energy_risk = []
        force_risk = []
        for structure_id in structure_ids:
            struct = frame[frame["structure_id"] == structure_id]
            if tier in {"A", "B"} and "router__target_valid_forces" in struct.columns:
                struct = struct[struct["router__target_valid_forces"] >= 1.0]
            elif "router__target_valid_energy" in struct.columns:
                struct = struct[struct["router__target_valid_energy"] >= 1.0]
            if struct.empty:
                continue
            candidate = struct.loc[struct[prob_col].idxmax()]
            if candidate[prob_col] < threshold:
                continue
            accepted += 1
            hit = int(candidate[label_col] == 1.0)
            hits += hit
            energy_risk.append(candidate["target__energy_err_pa"])
            force_risk.append(candidate["target__force_mean_err"])
        precision = hits / accepted if accepted else 0.0
        coverage = accepted / len(structure_ids)
        results.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "precision": precision,
                "accepted": accepted,
                "mean_energy_err": float(np.nanmean(energy_risk)) if energy_risk else float("nan"),
                "mean_force_err": float(np.nanmean(force_risk)) if force_risk else float("nan"),
            }
        )
    result_df = pd.DataFrame(results)
    valid = result_df[result_df["precision"] >= min_precision]
    if valid.empty:
        best = result_df.sort_values(["precision", "coverage"], ascending=[False, False]).iloc[0]
    else:
        best = valid.sort_values(["coverage", "precision"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), result_df


def apply_hierarchical_decision(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    tier_order: list[str],
) -> pd.DataFrame:
    decisions = []
    for structure_id, struct in frame.groupby("structure_id", sort=True):
        accepted = False
        for tier in tier_order:
            eligible = struct
            if tier in {"A", "B"} and "router__target_valid_forces" in eligible.columns:
                eligible = eligible[eligible["router__target_valid_forces"] >= 1.0]
            elif "router__target_valid_energy" in eligible.columns:
                eligible = eligible[eligible["router__target_valid_energy"] >= 1.0]
            if eligible.empty:
                continue
            prob_col = f"pred__tier_{tier}"
            idx = eligible[prob_col].idxmax()
            candidate = eligible.loc[idx]
            if candidate[prob_col] >= thresholds[tier]:
                decisions.append(
                    {
                        "structure_id": int(structure_id),
                        "immutable_id": candidate["immutable_id"],
                        "status": "accept",
                        "chosen_tier": tier,
                        "chosen_model": candidate["model_name"],
                        "confidence": float(candidate[prob_col]),
                        "target__tier_hit": float(candidate.get(f"target__tier_{tier}", np.nan)),
                        "target__energy_err_pa": candidate.get("target__energy_err_pa", np.nan),
                        "target__force_mean_err": candidate.get("target__force_mean_err", np.nan),
                        "target__force_p90_err": candidate.get("target__force_p90_err", np.nan),
                    }
                )
                accepted = True
                break
        if not accepted:
            fallback_pool = struct
            if "router__target_valid_energy" in fallback_pool.columns:
                available = fallback_pool[fallback_pool["router__target_valid_energy"] >= 1.0]
                if not available.empty:
                    fallback_pool = available
            best_idx = fallback_pool["pred__tier_B"].idxmax()
            fallback = fallback_pool.loc[best_idx]
            decisions.append(
                {
                    "structure_id": int(structure_id),
                    "immutable_id": fallback["immutable_id"],
                    "status": "reject",
                    "chosen_tier": "R",
                    "chosen_model": fallback["model_name"],
                    "confidence": float(fallback["pred__tier_B"]),
                    "target__tier_hit": float(fallback.get("target__tier_B", np.nan)),
                    "target__energy_err_pa": fallback.get("target__energy_err_pa", np.nan),
                    "target__force_mean_err": fallback.get("target__force_mean_err", np.nan),
                    "target__force_p90_err": fallback.get("target__force_p90_err", np.nan),
                }
            )
    return pd.DataFrame(decisions)


def summarize_decisions(decisions: pd.DataFrame) -> dict:
    accepted = decisions[decisions["status"] == "accept"]
    summary = {
        "num_structures": int(len(decisions)),
        "coverage": float(len(accepted) / len(decisions)) if len(decisions) else 0.0,
        "precision": float(accepted["target__tier_hit"].mean()) if len(accepted) else 0.0,
        "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
        "mean_force_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
        "mean_force_p90_err": float(accepted["target__force_p90_err"].mean()) if len(accepted) else float("nan"),
        "tier_counts": accepted["chosen_tier"].value_counts().to_dict(),
        "model_counts": accepted["chosen_model"].value_counts().to_dict(),
    }
    return summary


def build_risk_coverage_curve(
    frame: pd.DataFrame,
    base_thresholds: dict[str, float],
    tier_order: list[str],
) -> pd.DataFrame:
    rows = []
    for delta in np.linspace(-0.30, 0.30, 25):
        thresholds = {
            tier: float(np.clip(base_thresholds[tier] + delta, 0.01, 0.99)) for tier in tier_order
        }
        decisions = apply_hierarchical_decision(frame, thresholds, tier_order)
        summary = summarize_decisions(decisions)
        rows.append({"delta": float(delta), **summary})
    return pd.DataFrame(rows)


def recall_at_k(
    frame: pd.DataFrame,
    prob_col: str,
    label_col: str,
    ks: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, float]:
    counts = {k: 0 for k in ks}
    total = 0
    for _, struct in frame.groupby("structure_id", sort=True):
        ranked = struct.sort_values(prob_col, ascending=False)
        total += 1
        for k in ks:
            if ranked.head(k)[label_col].max() >= 1.0:
                counts[k] += 1
    return {f"recall@{k}": counts[k] / total for k in ks}


def save_artifact_bundle(path: Path, bundle: dict) -> None:
    joblib.dump(bundle, path)


def load_artifact_bundle(path: Path) -> dict:
    return joblib.load(path)
