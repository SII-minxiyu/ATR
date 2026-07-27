from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, r2_score, roc_auc_score

from .constants import DEFAULT_TIERS
from .data import load_labeled_records, load_unlabeled_records
from .features import build_pair_table
from .modeling import (
    SplitBundle,
    add_ood_features,
    assign_clusters,
    fit_binary_classifier,
    load_artifact_bundle,
    predict_probabilities,
    save_artifact_bundle,
    split_groups,
)
from .pipeline import (
    _augment_inference_features,
    _router_feature_cols,
    _single_record_pair_frame,
    _structure_feature_cols,
    resolve_disagreement_models,
)


LIGHT_POSTPROCESS_CONFIDENCE_LT = 0.50
LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT = 0.10
FORCE_ZOOM_RANGE = 20.0


def _group_kfold_splits(
    pair_df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[set[int], set[int]]]:
    groups = pair_df[["structure_id", "group_id"]].drop_duplicates()
    unique_groups = groups["group_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    folded_groups = np.array_split(unique_groups, n_splits)

    splits = []
    for fold_idx in range(n_splits):
        val_groups = set(folded_groups[fold_idx].tolist())
        train_groups = set()
        for other_idx, bucket in enumerate(folded_groups):
            if other_idx == fold_idx:
                continue
            train_groups.update(bucket.tolist())

        train_ids = set(groups[groups["group_id"].isin(train_groups)]["structure_id"].astype(int).tolist())
        val_ids = set(groups[groups["group_id"].isin(val_groups)]["structure_id"].astype(int).tolist())
        splits.append((train_ids, val_ids))
    return splits


def _structure_level_threshold_search(
    frame: pd.DataFrame,
    prob_col: str,
    label_col: str,
    min_precision: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    thresholds = np.linspace(0.05, 0.99, 95)
    n_structures = frame["structure_id"].nunique()
    for threshold in thresholds:
        decisions = _select_best_model(frame, prob_col=prob_col, threshold=threshold)
        accepted = decisions[decisions["status"] == "accept"].copy()
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(len(accepted) / n_structures) if n_structures else 0.0,
                "precision": float(accepted[label_col].mean()) if len(accepted) else 0.0,
                "accepted": int(len(accepted)),
                "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
                "mean_force_mean_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
                "mean_force_max_err": float(accepted["target__force_max_err"].mean()) if len(accepted) else float("nan"),
            }
        )
    result_df = pd.DataFrame(rows)
    valid = result_df[result_df["precision"] >= min_precision]
    if valid.empty:
        best = result_df.sort_values(["precision", "coverage"], ascending=[False, False]).iloc[0]
    else:
        best = valid.sort_values(["coverage", "precision"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), result_df


def _select_best_model(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    threshold: float,
) -> pd.DataFrame:
    decisions = []
    for structure_id, struct in frame.groupby("structure_id", sort=True):
        idx = struct[prob_col].idxmax()
        best = struct.loc[idx]
        accept = float(best[prob_col]) >= threshold
        target_tier = float(best["target__tier_A"]) if "target__tier_A" in best.index else float("nan")
        target_energy = (
            float(best["target__energy_err_pa"]) if "target__energy_err_pa" in best.index else float("nan")
        )
        target_force_mean = (
            float(best["target__force_mean_err"]) if "target__force_mean_err" in best.index else float("nan")
        )
        target_force_max = (
            float(best["target__force_max_err"]) if "target__force_max_err" in best.index else float("nan")
        )
        decisions.append(
            {
                "structure_id": int(structure_id),
                "immutable_id": best["immutable_id"],
                "split": best.get("split", ""),
                "status": "accept" if accept else "reject",
                "chosen_label": best["model_name"] if accept else "reject",
                "chosen_model": best["model_name"] if accept else None,
                "confidence": float(best[prob_col]),
                "target__tier_A": target_tier,
                "target__energy_err_pa": target_energy,
                "target__force_mean_err": target_force_mean,
                "target__force_max_err": target_force_max,
            }
        )
    return pd.DataFrame(decisions)


def _merge_selected_pair_features(decisions: pd.DataFrame, pair_frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "structure_id",
        "model_name",
        "router__target_force_dev_mean",
        "router__target_force_dev_p90",
        "router__target_force_dev_max",
        "router__target_force_component_abs_ratio_mean",
        "router__target_force_component_abs_ratio_p10",
        "router__target_force_component_abs_ratio_p90",
        "router__target_force_component_abs_diff_mean",
        "router__target_force_component_abs_diff_p90",
        "router__target_force_component_sign_mismatch_ratio",
        "router__target_force_component_compress_count_proxy",
        "router__target_force_component_compress_ratio_proxy",
        "router__target_force_component_near_zero_ratio_proxy",
        "router__target_force_active_component_frac",
    ]
    available_cols = [col for col in feature_cols if col in pair_frame.columns]
    selected = pair_frame[available_cols].copy()
    merged = decisions.merge(
        selected,
        left_on=["structure_id", "chosen_model"],
        right_on=["structure_id", "model_name"],
        how="left",
    )
    if "model_name" in merged.columns:
        merged = merged.drop(columns=["model_name"])
    return merged


def apply_light_postprocess(
    decisions: pd.DataFrame,
    pair_frame: pd.DataFrame,
    *,
    confidence_lt: float = LIGHT_POSTPROCESS_CONFIDENCE_LT,
    force_dev_max_gt: float = LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
) -> pd.DataFrame:
    post = _merge_selected_pair_features(decisions.copy(), pair_frame)
    post["pre_postprocess_status"] = post["status"]
    post["pre_postprocess_label"] = post["chosen_label"]
    post["pre_postprocess_model"] = post["chosen_model"]
    post["postprocess_reject"] = False
    post["postprocess_reason"] = ""

    reject_mask = (
        (post["status"] == "accept")
        & (post["confidence"] < confidence_lt)
        & (post["router__target_force_dev_max"] > force_dev_max_gt)
    )

    post.loc[reject_mask, "status"] = "reject"
    post.loc[reject_mask, "chosen_label"] = "reject"
    post.loc[reject_mask, "chosen_model"] = None
    post.loc[reject_mask, "postprocess_reject"] = True
    post.loc[
        reject_mask,
        "postprocess_reason",
    ] = f"confidence<{confidence_lt:.2f} and force_dev_max>{force_dev_max_gt:.2f}"
    return post


def _pair_full_export(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cols = [
        "structure_id",
        "immutable_id",
        "group_id",
        "model_name",
        "structure__natoms",
        "structure__unique_elements",
        "structure__volume_per_atom",
        "structure__density",
        "structure__ood_knn_mean",
        "structure__cluster_id",
        "router__energy_ens_std",
        "router__energy_pair_mean",
        "router__energy_pair_max",
        "router__force_pair_mean",
        "router__force_pair_max",
        "router__target_energy_pa",
        "router__target_force_mean",
        "router__target_energy_dev_to_median",
        "router__target_force_dev_mean",
        "target__tier_A",
        "target__energy_err_pa",
        "target__force_mean_err",
        "target__force_max_err",
        "pred__tier_A",
        "split",
    ]
    out = frame[cols].copy()
    out = out.rename(
        columns={
            "model_name": "teacher_model",
            "structure__natoms": "natoms",
            "structure__unique_elements": "n_elem",
            "structure__volume_per_atom": "vol_pa",
            "structure__density": "density",
            "structure__ood_knn_mean": "ood",
            "structure__cluster_id": "cluster_id",
            "router__energy_ens_std": "energy_std_ensemble",
            "router__energy_pair_mean": "energy_pair_mean_ensemble",
            "router__energy_pair_max": "energy_pair_max_ensemble",
            "router__force_pair_mean": "force_pair_mean_ensemble",
            "router__force_pair_max": "force_pair_max_ensemble",
            "router__target_energy_pa": "teacher_energy_pa",
            "router__target_force_mean": "teacher_force_mean",
            "router__target_energy_dev_to_median": "this_teacher_energy_dev",
            "router__target_force_dev_mean": "this_teacher_force_dev",
            "target__tier_A": "label_A",
            "target__energy_err_pa": "energy_err_pa",
            "target__force_mean_err": "force_mean_err",
            "target__force_max_err": "force_max_err",
            "pred__tier_A": "pred_prob_A",
        }
    )
    out["split"] = split_name
    return out


def _probability_metrics(frame: pd.DataFrame, label_col: str, prob_col: str) -> dict:
    y_true = frame[label_col].to_numpy(dtype=float)
    y_prob = frame[prob_col].to_numpy(dtype=float)
    metrics = {
        "r2_prob_vs_label": float(r2_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0.0, 1.0])),
        "brier_loss": float(brier_score_loss(y_true, y_prob)),
    }
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    if positives > 0 and negatives > 0:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def route_ar_structures(
    bundle_path: Path,
    structure_db: Path,
    model_dbs: list[Path],
    structure_ids: list[int] | None = None,
    threshold: float | None = None,
    *,
    postprocess_light: bool = False,
    postprocess_confidence_lt: float = LIGHT_POSTPROCESS_CONFIDENCE_LT,
    postprocess_force_dev_max_gt: float = LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
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

    if not frames:
        return pd.DataFrame(
            columns=[
                "structure_id",
                "immutable_id",
                "status",
                "chosen_label",
                "chosen_model",
                "confidence",
                "threshold",
            ]
        )

    all_pairs = pd.concat(frames, ignore_index=True)
    all_pairs = _augment_inference_features(all_pairs, bundle)
    prob_col = "pred__tier_A"
    all_pairs[prob_col] = bundle["router_model"].predict_proba(
        all_pairs[bundle["router_feature_cols"]].to_numpy()
    )[:, 1]

    decision_threshold = float(threshold if threshold is not None else bundle["threshold"])
    decisions = _select_best_model(all_pairs, prob_col=prob_col, threshold=decision_threshold)
    decisions["threshold"] = decision_threshold

    score_cols = ["structure_id", "immutable_id", "model_name", prob_col]
    ranked_pairs = all_pairs[score_cols].copy().sort_values(["structure_id", prob_col], ascending=[True, False])
    top3 = ranked_pairs.groupby("structure_id").head(3).copy()
    top3["rank"] = top3.groupby("structure_id").cumcount() + 1
    pivot = top3.pivot(index="structure_id", columns="rank", values=["model_name", prob_col])
    pivot.columns = [f"top{rank}_{'model' if field == 'model_name' else 'score'}" for field, rank in pivot.columns]
    pivot = pivot.reset_index()
    decisions = decisions.merge(pivot, on="structure_id", how="left")
    if postprocess_light:
        decisions = apply_light_postprocess(
            decisions,
            all_pairs,
            confidence_lt=postprocess_confidence_lt,
            force_dev_max_gt=postprocess_force_dev_max_gt,
        )
    return decisions.sort_values("structure_id").reset_index(drop=True)


def run_ar_maxforce_cross_validation(
    root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    backend: str = "xgboost",
    disagreement_mode: str = "all",
    precision_target: float = 0.90,
    n_clusters: int = 12,
    n_splits: int = 5,
    calibration_fraction: float = 0.125,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    records, model_names = load_labeled_records(root)
    disagreement_models = resolve_disagreement_models(disagreement_mode, None)
    pair_df = build_pair_table(
        records,
        model_names,
        DEFAULT_TIERS,
        disagreement_models=disagreement_models,
    )
    pair_df["target__tier_A"] = (
        (pair_df["target__valid_energy"] >= 1.0)
        & (pair_df["target__valid_forces"] >= 1.0)
        & (pair_df["target__energy_err_pa"] < 0.10)
        & (pair_df["target__force_max_err"] < 1.0)
    ).astype(float)

    outer_splits = _group_kfold_splits(pair_df, n_splits=n_splits, seed=seed)
    fold_summaries = []
    all_decisions = []

    for fold_idx, (outer_train_ids, outer_val_ids) in enumerate(outer_splits, start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        outer_train_frame = pair_df[pair_df["structure_id"].isin(outer_train_ids)].copy()
        train_split = split_groups(
            outer_train_frame,
            train_fraction=max(0.50, 1.0 - calibration_fraction),
            val_fraction=min(0.49, calibration_fraction),
            seed=seed + fold_idx,
        )
        calib_ids = set(train_split.val_ids)
        train_ids = set(train_split.train_ids) | set(train_split.test_ids)

        split_map = {}
        for structure_id in train_ids:
            split_map[int(structure_id)] = "train"
        for structure_id in calib_ids:
            split_map[int(structure_id)] = "calib"
        for structure_id in outer_val_ids:
            split_map[int(structure_id)] = "val"

        fold_pair_df = pair_df[
            pair_df["structure_id"].isin(train_ids | calib_ids | outer_val_ids)
        ].copy()
        fold_pair_df["split"] = fold_pair_df["structure_id"].map(split_map).fillna("unused")

        structure_cols = _structure_feature_cols(fold_pair_df)
        split_bundle = SplitBundle(
            train_ids=train_ids,
            val_ids=calib_ids,
            test_ids=outer_val_ids,
        )
        fold_pair_df, _ = add_ood_features(fold_pair_df, split_bundle, structure_cols)
        fold_pair_df, _ = assign_clusters(
            fold_pair_df,
            split_bundle,
            structure_cols,
            n_clusters=n_clusters,
            seed=seed + fold_idx,
        )
        feature_cols = _router_feature_cols(fold_pair_df)

        train_frame = fold_pair_df[fold_pair_df["structure_id"].isin(train_ids)].copy()
        calib_frame = fold_pair_df[fold_pair_df["structure_id"].isin(calib_ids)].copy()
        val_frame = fold_pair_df[fold_pair_df["structure_id"].isin(outer_val_ids)].copy()

        model, model_info = fit_binary_classifier(
            train_frame,
            feature_cols,
            "target__tier_A",
            seed=seed + fold_idx,
            backend=backend,
        )
        for frame in (train_frame, calib_frame, val_frame):
            frame["pred__tier_A"] = predict_probabilities(model, frame, feature_cols)

        pair_metrics = _probability_metrics(val_frame, "target__tier_A", "pred__tier_A")

        threshold, threshold_table = _structure_level_threshold_search(
            calib_frame,
            prob_col="pred__tier_A",
            label_col="target__tier_A",
            min_precision=precision_target,
        )
        threshold_table.to_csv(fold_dir / "threshold_search_A.csv", index=False)

        val_decisions = _select_best_model(val_frame, prob_col="pred__tier_A", threshold=threshold)
        val_decisions["fold"] = fold_idx
        val_decisions.to_csv(fold_dir / "val_structure_decisions.csv", index=False)
        all_decisions.append(val_decisions)

        accepted = val_decisions[val_decisions["status"] == "accept"].copy()
        fold_summary = {
            "fold": int(fold_idx),
            "threshold": float(threshold),
            "backend": backend,
            "backend_positive_rate": float(model_info["positive_rate"]),
            "train_structures": int(len(train_ids)),
            "calib_structures": int(len(calib_ids)),
            "val_structures": int(len(outer_val_ids)),
            "coverage": float(len(accepted) / len(val_decisions)) if len(val_decisions) else 0.0,
            "precision": float(accepted["target__tier_A"].mean()) if len(accepted) else 0.0,
            "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
            "mean_force_mean_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
            "mean_force_max_err": float(accepted["target__force_max_err"].mean()) if len(accepted) else float("nan"),
            "accept_count": int(len(accepted)),
            "reject_count": int((val_decisions["status"] == "reject").sum()),
            "pair_r2": float(pair_metrics["r2_prob_vs_label"]),
            "pair_log_loss": float(pair_metrics["log_loss"]),
            "pair_brier_loss": float(pair_metrics["brier_loss"]),
            "pair_roc_auc": float(pair_metrics["roc_auc"]),
            "pair_pr_auc": float(pair_metrics["pr_auc"]),
        }
        fold_summaries.append(fold_summary)

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    all_decisions_df = pd.concat(all_decisions, ignore_index=True).sort_values(["structure_id", "fold"])
    all_decisions_df.to_csv(output_dir / "oof_structure_decisions.csv", index=False)

    accepted_all = all_decisions_df[all_decisions_df["status"] == "accept"].copy()
    overall = {
        "num_structures": int(all_decisions_df["structure_id"].nunique()),
        "coverage": float(len(accepted_all) / len(all_decisions_df)) if len(all_decisions_df) else 0.0,
        "precision": float(accepted_all["target__tier_A"].mean()) if len(accepted_all) else 0.0,
        "mean_energy_err_pa": float(accepted_all["target__energy_err_pa"].mean()) if len(accepted_all) else float("nan"),
        "mean_force_mean_err": float(accepted_all["target__force_mean_err"].mean()) if len(accepted_all) else float("nan"),
        "mean_force_max_err": float(accepted_all["target__force_max_err"].mean()) if len(accepted_all) else float("nan"),
        "accept_count": int(len(accepted_all)),
        "reject_count": int((all_decisions_df["status"] == "reject").sum()),
    }

    metric_cols = [
        "threshold",
        "coverage",
        "precision",
        "mean_energy_err_pa",
        "mean_force_mean_err",
        "mean_force_max_err",
        "accept_count",
        "reject_count",
        "pair_r2",
        "pair_log_loss",
        "pair_brier_loss",
        "pair_roc_auc",
        "pair_pr_auc",
    ]
    aggregate = {}
    for col in metric_cols:
        values = fold_df[col].to_numpy(dtype=float)
        aggregate[col] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }

    summary = {
        "experiment": "A_or_R_max_force_5fold_cv",
        "definition": {
            "A": {
                "energy_err_pa_lt": 0.10,
                "force_max_err_lt": 1.0,
            },
            "R": "otherwise",
        },
        "backend": backend,
        "disagreement_mode": disagreement_mode,
        "disagreement_models": disagreement_models,
        "n_splits": int(n_splits),
        "calibration_fraction_within_train": float(calibration_fraction),
        "precision_target": float(precision_target),
        "fold_metrics": fold_summaries,
        "aggregate": aggregate,
        "overall_oof": overall,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _export_exact_model_io(
    output_dir: Path,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    io_dir = output_dir / "model_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    def write_split(frame: pd.DataFrame, split_name: str) -> None:
        meta_cols = ["structure_id", "immutable_id", "group_id", "model_name", "split"]
        x = pd.concat([frame[meta_cols].copy(), frame[feature_cols].copy()], axis=1)
        y = frame[
            [
                "structure_id",
                "immutable_id",
                "group_id",
                "model_name",
                "split",
                "target__tier_A",
                "target__energy_err_pa",
                "target__force_mean_err",
                "target__force_max_err",
                "pred__tier_A",
            ]
        ].copy()
        y = y.rename(
            columns={
                "target__tier_A": "label_A",
                "target__energy_err_pa": "energy_err_pa",
                "target__force_mean_err": "force_mean_err",
                "target__force_max_err": "force_max_err",
                "pred__tier_A": "pred_prob_A",
            }
        )
        x.to_csv(io_dir / f"{split_name}_inputs_exact.csv", index=False)
        y.to_csv(io_dir / f"{split_name}_targets_exact.csv", index=False)
        merged = pd.concat([x, y[["label_A", "energy_err_pa", "force_mean_err", "force_max_err", "pred_prob_A"]]], axis=1)
        merged.to_csv(io_dir / f"{split_name}_dataset_merged.csv", index=False)

    write_split(train_frame, "train")
    write_split(val_frame, "val")
    (io_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))


def _build_choice_outputs(
    decisions: pd.DataFrame,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    labels = ["reject", *model_names]
    code_map = {label: idx for idx, label in enumerate(labels)}

    code_rows = []
    bool_rows = []
    for row in decisions.itertuples(index=False):
        chosen_label = row.chosen_label
        is_misclassified = (
            (row.status == "accept" and float(row.target__tier_A) < 1.0)
            or (row.status == "reject" and float(row.target__tier_A) >= 1.0)
        )
        code_rows.append(
            {
                "structure_id": int(row.structure_id),
                "immutable_id": row.immutable_id,
                "split": row.split,
                "choice_code": int(code_map[chosen_label]),
                "choice_label": chosen_label,
                "status": row.status,
                "confidence": float(row.confidence),
            }
        )
        bool_row = {
            "structure_id": int(row.structure_id),
            "immutable_id": row.immutable_id,
            "split": row.split,
            "misclassified": True if is_misclassified else "",
        }
        for label in labels:
            bool_row[label] = bool(chosen_label == label)
        bool_rows.append(bool_row)

    return pd.DataFrame(code_rows), pd.DataFrame(bool_rows), code_map


def _collect_parity_frames(records, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = decisions[decisions["status"] == "accept"].copy()
    chosen = dict(zip(accepted["structure_id"], accepted["chosen_model"]))
    confidence = dict(zip(accepted["structure_id"], accepted["confidence"]))
    force_dev_max = dict(
        zip(
            accepted["structure_id"],
            accepted.get("router__target_force_dev_max", pd.Series([np.nan] * len(accepted))),
        )
    )

    energy_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    for record in records:
        model_name = chosen.get(record.structure_id)
        if model_name is None:
            continue
        pred = record.predictions[model_name]
        if pred.energy is not None and record.dft_energy is not None:
            energy_rows.append(
                {
                    "structure_id": int(record.structure_id),
                    "immutable_id": record.immutable_id,
                    "chosen_model": model_name,
                    "confidence": float(confidence[record.structure_id]),
                    "e_dft": float(record.dft_energy / record.natoms),
                    "e_mlff": float(pred.energy / record.natoms),
                }
            )
        if pred.forces is not None and record.dft_forces is not None:
            f_dft = record.dft_forces.reshape(-1)
            f_mlff = pred.forces.reshape(-1)
            for fd, fm in zip(f_dft, f_mlff):
                force_rows.append(
                    {
                        "structure_id": int(record.structure_id),
                        "immutable_id": record.immutable_id,
                        "chosen_model": model_name,
                        "confidence": float(confidence[record.structure_id]),
                        "router__target_force_dev_max": float(force_dev_max[record.structure_id]),
                        "f_dft": float(fd),
                        "f_mlff": float(fm),
                    }
                )
    return pd.DataFrame(energy_rows), pd.DataFrame(force_rows)


def _parity_metrics(frame: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    diff = frame[y_col].to_numpy() - frame[x_col].to_numpy()
    y = frame[x_col].to_numpy()
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(diff**2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _plot_energy_force_parity(
    energy_df: pd.DataFrame,
    force_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    energy_metrics = _parity_metrics(energy_df, "e_dft", "e_mlff")
    force_metrics = _parity_metrics(force_df, "f_dft", "f_mlff")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

    ax = axes[0]
    ax.scatter(energy_df["e_dft"], energy_df["e_mlff"], s=8, alpha=0.5, c="#2a6fdb", edgecolors="none", rasterized=True)
    elo = float(min(energy_df["e_dft"].min(), energy_df["e_mlff"].min()))
    ehi = float(max(energy_df["e_dft"].max(), energy_df["e_mlff"].max()))
    epad = 0.02 * (ehi - elo)
    ax.plot([elo - epad, ehi + epad], [elo - epad, ehi + epad], "k--", linewidth=1.5)
    ax.set_xlim(elo - epad, ehi + epad)
    ax.set_ylim(elo - epad, ehi + epad)
    ax.set_xlabel(r"$E_{\mathrm{DFT}}$ [eV/atom]")
    ax.set_ylabel(r"$E_{\mathrm{MLFF}}$ [eV/atom]")
    ax.set_title("Energy parity")
    ax.grid(alpha=0.15)
    ax.text(
        0.03,
        0.97,
        f"MAE = {energy_metrics['mae']:.3f}, RMSE = {energy_metrics['rmse']:.3f}, $R^2$ = {energy_metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    ax = axes[1]
    ax.scatter(force_df["f_dft"], force_df["f_mlff"], s=4, alpha=0.18, c="#2a6fdb", edgecolors="none", rasterized=True)
    flo = float(min(force_df["f_dft"].min(), force_df["f_mlff"].min()))
    fhi = float(max(force_df["f_dft"].max(), force_df["f_mlff"].max()))
    fpad = 0.02 * (fhi - flo)
    ax.plot([flo - fpad, fhi + fpad], [flo - fpad, fhi + fpad], "k--", linewidth=1.5)
    ax.set_xlim(flo - fpad, fhi + fpad)
    ax.set_ylim(flo - fpad, fhi + fpad)
    ax.set_xlabel(r"$F_{\mathrm{DFT}}$ [eV/$\AA$]")
    ax.set_ylabel(r"$F_{\mathrm{MLFF}}$ [eV/$\AA$]")
    ax.set_title("Force parity")
    ax.grid(alpha=0.15)
    ax.text(
        0.03,
        0.97,
        f"MAE = {force_metrics['mae']:.3f}, RMSE = {force_metrics['rmse']:.3f}, $R^2$ = {force_metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return {"energy": energy_metrics, "force": force_metrics}


def _plot_force_zoom(force_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    zoom = force_df[
        (force_df["f_dft"].between(-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE))
        & (force_df["f_mlff"].between(-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE))
    ].copy()
    ax.scatter(zoom["f_dft"], zoom["f_mlff"], s=6, alpha=0.22, c="#2a6fdb", edgecolors="none", rasterized=True)
    ax.plot(
        [-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE],
        [-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE],
        "k--",
        linewidth=1.5,
    )
    ax.set_xlim(-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE)
    ax.set_ylim(-FORCE_ZOOM_RANGE, FORCE_ZOOM_RANGE)
    ax.set_xlabel(r"$F_{\mathrm{DFT}}$ [eV/$\AA$]")
    ax.set_ylabel(r"$F_{\mathrm{MLFF}}$ [eV/$\AA$]")
    ax.set_title("Accepted structures only (zoom to +/-20)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _compression_anomaly_frame(force_df: pd.DataFrame) -> pd.DataFrame:
    out = force_df.copy()
    out["abs_f_dft"] = np.abs(out["f_dft"])
    out["abs_f_mlff"] = np.abs(out["f_mlff"])
    out["abs_diff"] = np.abs(out["f_mlff"] - out["f_dft"])
    out["compression_anomaly"] = (
        (out["abs_f_dft"] >= 2.0)
        & (out["abs_f_dft"] <= 20.0)
        & (out["abs_f_mlff"] <= 0.7 * out["abs_f_dft"])
        & (out["abs_diff"] >= 2.0)
    )
    return out


def run_ar_maxforce_experiment(
    root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    backend: str = "xgboost",
    disagreement_mode: str = "all",
    precision_target: float = 0.90,
    n_clusters: int = 12,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    records, model_names = load_labeled_records(root)
    disagreement_models = resolve_disagreement_models(disagreement_mode, None)
    pair_df = build_pair_table(
        records,
        model_names,
        DEFAULT_TIERS,
        disagreement_models=disagreement_models,
    )

    # Overwrite the label definition for this experiment:
    # A if |dE| < 0.1 eV/atom and max atomic force error < 1.0 eV/A, else R.
    pair_df["target__tier_A"] = (
        (pair_df["target__valid_energy"] >= 1.0)
        & (pair_df["target__valid_forces"] >= 1.0)
        & (pair_df["target__energy_err_pa"] < 0.10)
        & (pair_df["target__force_max_err"] < 1.0)
    ).astype(float)

    split = split_groups(pair_df, train_fraction=0.80, val_fraction=0.20, seed=seed)
    split_map = {}
    for structure_id in split.train_ids:
        split_map[int(structure_id)] = "train"
    for structure_id in split.val_ids:
        split_map[int(structure_id)] = "val"
    pair_df["split"] = pair_df["structure_id"].map(split_map).fillna("unused")

    structure_cols = _structure_feature_cols(pair_df)
    pair_df, ood_artifact = add_ood_features(pair_df, split, structure_cols)
    pair_df, cluster_artifact = assign_clusters(pair_df, split, structure_cols, n_clusters=n_clusters, seed=seed)
    feature_cols = _router_feature_cols(pair_df)

    train_frame = pair_df[pair_df["structure_id"].isin(split.train_ids)].copy()
    val_frame = pair_df[pair_df["structure_id"].isin(split.val_ids)].copy()
    all_frame = pair_df[pair_df["structure_id"].isin(split.train_ids | split.val_ids)].copy()

    model, model_info = fit_binary_classifier(
        train_frame,
        feature_cols,
        "target__tier_A",
        seed=seed,
        backend=backend,
    )
    for frame in (train_frame, val_frame, all_frame):
        frame["pred__tier_A"] = predict_probabilities(model, frame, feature_cols)

    threshold, threshold_table = _structure_level_threshold_search(
        val_frame,
        prob_col="pred__tier_A",
        label_col="target__tier_A",
        min_precision=precision_target,
    )
    threshold_table.to_csv(output_dir / "threshold_search_A.csv", index=False)

    _export_exact_model_io(output_dir, train_frame, val_frame, feature_cols)

    train_pairs_full = _pair_full_export(train_frame, "train")
    val_pairs_full = _pair_full_export(val_frame, "val")
    train_pairs_full.to_csv(output_dir / "train_labels.csv", index=False)
    val_pairs_full.to_csv(output_dir / "val_labels.csv", index=False)

    for stale_name in ("train_pairs_full.csv", "val_pairs_full.csv"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    val_decisions = _select_best_model(val_frame, prob_col="pred__tier_A", threshold=threshold)
    all_decisions = _select_best_model(all_frame, prob_col="pred__tier_A", threshold=threshold)
    val_decisions.to_csv(output_dir / "val_structure_decisions.csv", index=False)
    all_decisions.to_csv(output_dir / "all_structure_decisions.csv", index=False)

    code_df, bool_df, code_map = _build_choice_outputs(all_decisions, model_names)
    code_df.to_csv(output_dir / "structure_choice_code.csv", index=False)
    bool_df.to_csv(output_dir / "structure_choice_bool_matrix.csv", index=False)
    (output_dir / "choice_code_mapping.json").write_text(json.dumps(code_map, indent=2))

    oracle_best = (
        pair_df.groupby("structure_id", as_index=False)["target__tier_A"]
        .max()
        .rename(columns={"target__tier_A": "oracle_hit"})
    )

    def summarize(decisions: pd.DataFrame) -> dict:
        accepted = decisions[decisions["status"] == "accept"].copy()
        n_structures = len(decisions)
        return {
            "num_structures": int(n_structures),
            "coverage": float(len(accepted) / n_structures) if n_structures else 0.0,
            "precision": float(accepted["target__tier_A"].mean()) if len(accepted) else 0.0,
            "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
            "mean_force_mean_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
            "mean_force_max_err": float(accepted["target__force_max_err"].mean()) if len(accepted) else float("nan"),
            "accept_count": int(len(accepted)),
            "reject_count": int((decisions["status"] == "reject").sum()),
            "model_counts": accepted["chosen_label"].value_counts().to_dict(),
        }

    split_summary = (
        pair_df[["structure_id", "split"]]
        .drop_duplicates("structure_id")
        .groupby("split")
        .size()
        .to_dict()
    )
    label_summary = (
        pair_df.groupby("split", as_index=False)["target__tier_A"]
        .mean()
        .rename(columns={"target__tier_A": "positive_rate"})
        .to_dict(orient="records")
    )

    bundle = {
        "experiment": "A_or_R_max_force",
        "seed": seed,
        "backend": backend,
        "model_names": model_names,
        "disagreement_mode": disagreement_mode,
        "disagreement_models": disagreement_models,
        "definition": {
            "A": {
                "energy_err_pa_lt": 0.10,
                "force_max_err_lt": 1.0,
            },
            "R": "otherwise",
        },
        "router_feature_cols": feature_cols,
        "router_model": model,
        "threshold": float(threshold),
        "ood_artifact": ood_artifact,
        "cluster_artifact": cluster_artifact,
    }
    save_artifact_bundle(output_dir / "router_bundle.joblib", bundle)

    summary = {
        "experiment": "A_or_R_max_force",
        "definition": {
            "A": {
                "energy_err_pa_lt": 0.10,
                "force_max_err_lt": 1.0,
            },
            "R": "otherwise",
        },
        "split": {
            "train_fraction": 0.80,
            "val_fraction": 0.20,
            "structure_counts": split_summary,
        },
        "backend": backend,
        "backend_info": model_info,
        "disagreement_mode": disagreement_mode,
        "disagreement_models": disagreement_models,
        "threshold": threshold,
        "precision_target": precision_target,
        "oracle": {
            "coverage": float(oracle_best["oracle_hit"].mean()),
            "accept_count": int(oracle_best["oracle_hit"].sum()),
        },
        "label_summary": label_summary,
        "pair_metrics": {
            "train": _probability_metrics(train_frame, "target__tier_A", "pred__tier_A"),
            "val": _probability_metrics(val_frame, "target__tier_A", "pred__tier_A"),
        },
        "validation": summarize(val_decisions),
        "all_labeled": summarize(all_decisions),
    }
    (output_dir / "pair_metrics.json").write_text(json.dumps(summary["pair_metrics"], indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_ar_postprocess_analysis(
    root: Path,
    source_dir: Path,
    output_dir: Path,
    *,
    confidence_lt: float = LIGHT_POSTPROCESS_CONFIDENCE_LT,
    force_dev_max_gt: float = LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = source_dir / "summary.json"
    decisions_path = source_dir / "all_structure_decisions.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source summary: {summary_path}")
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing source decisions: {decisions_path}")

    source_summary = json.loads(summary_path.read_text())
    records, model_names = load_labeled_records(root)
    disagreement_models = source_summary.get("disagreement_models")
    pair_df = build_pair_table(records, model_names, DEFAULT_TIERS, disagreement_models=disagreement_models)
    pair_df["target__tier_A"] = (
        (pair_df["target__valid_energy"] >= 1.0)
        & (pair_df["target__valid_forces"] >= 1.0)
        & (pair_df["target__energy_err_pa"] < 0.10)
        & (pair_df["target__force_max_err"] < 1.0)
    ).astype(float)

    source_decisions = pd.read_csv(decisions_path)
    post_decisions = apply_light_postprocess(
        source_decisions,
        pair_df,
        confidence_lt=confidence_lt,
        force_dev_max_gt=force_dev_max_gt,
    ).sort_values("structure_id").reset_index(drop=True)

    post_decisions.to_csv(output_dir / "all_structure_decisions.csv", index=False)

    code_df, bool_df, code_map = _build_choice_outputs(post_decisions, model_names)
    code_df.to_csv(output_dir / "structure_choice_code.csv", index=False)
    bool_df.to_csv(output_dir / "structure_choice_bool_matrix.csv", index=False)
    (output_dir / "choice_code_mapping.json").write_text(json.dumps(code_map, indent=2))

    energy_df, force_df = _collect_parity_frames(records, post_decisions)
    energy_df.to_csv(output_dir / "accepted_energy_points.csv", index=False)
    force_df.to_csv(output_dir / "accepted_force_components.csv", index=False)

    parity_metrics = _plot_energy_force_parity(
        energy_df,
        force_df,
        output_dir / "energy_force_parity.png",
    )
    _plot_force_zoom(force_df, output_dir / "force_zoom_pm20.png")

    force_with_flags = _compression_anomaly_frame(force_df)
    compression = force_with_flags[force_with_flags["compression_anomaly"]].copy()
    compression.to_csv(output_dir / "compression_anomaly_components.csv", index=False)
    compression_summary = (
        compression.groupby(["structure_id", "immutable_id", "chosen_model", "confidence"], as_index=False)
        .size()
        .rename(columns={"size": "compression_anomaly_component_count"})
        .sort_values(["compression_anomaly_component_count", "confidence"], ascending=[False, True])
    )
    compression_summary.to_csv(output_dir / "compression_anomaly_structures.csv", index=False)

    before_accept = source_decisions[source_decisions["status"] == "accept"].copy()
    after_accept = post_decisions[post_decisions["status"] == "accept"].copy()
    post_rejected = post_decisions[post_decisions["postprocess_reject"]].copy()

    summary = {
        "source_dir": str(source_dir),
        "postprocess_rule": {
            "name": "light_force_deviation",
            "confidence_lt": float(confidence_lt),
            "force_dev_max_gt": float(force_dev_max_gt),
        },
        "before": {
            "accept_count": int(len(before_accept)),
            "reject_count": int((source_decisions["status"] == "reject").sum()),
            "precision": float(before_accept["target__tier_A"].mean()) if len(before_accept) else float("nan"),
        },
        "after": {
            "accept_count": int(len(after_accept)),
            "reject_count": int((post_decisions["status"] == "reject").sum()),
            "precision": float(after_accept["target__tier_A"].mean()) if len(after_accept) else float("nan"),
            "energy_mae": float(parity_metrics["energy"]["mae"]),
            "energy_rmse": float(parity_metrics["energy"]["rmse"]),
            "energy_r2": float(parity_metrics["energy"]["r2"]),
            "force_mae": float(parity_metrics["force"]["mae"]),
            "force_rmse": float(parity_metrics["force"]["rmse"]),
            "force_r2": float(parity_metrics["force"]["r2"]),
        },
        "postprocess_rejected_count": int(len(post_rejected)),
        "postprocess_rejected_true_positive_count": int(post_rejected["target__tier_A"].sum()),
        "postprocess_rejected_false_positive_count": int(len(post_rejected) - post_rejected["target__tier_A"].sum()),
        "compression_anomaly": {
            "component_count": int(len(compression)),
            "structure_count": int(compression["structure_id"].nunique()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
