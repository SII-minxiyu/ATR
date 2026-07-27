from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_TIERS
from .data import load_labeled_records
from .features import build_pair_table
from .modeling import apply_hierarchical_decision, load_artifact_bundle, summarize_decisions
from .pipeline import TIER_ORDER, _augment_inference_features


@dataclass
class PolicySpec:
    name: str
    tier_order: list[str]
    thresholds: dict[str, float]
    description: str


def _split_column(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    split_map = {}
    for split_name, ids in bundle["split"].items():
        mapped = split_name.replace("_ids", "")
        for structure_id in ids:
            split_map[int(structure_id)] = mapped
    out = frame.copy()
    out["split"] = out["structure_id"].map(split_map)
    return out


def build_scored_pair_frame(root: Path, bundle_path: Path) -> tuple[pd.DataFrame, dict]:
    bundle = load_artifact_bundle(bundle_path)
    records, model_names = load_labeled_records(root)
    pair_df = build_pair_table(
        records,
        model_names,
        DEFAULT_TIERS,
        disagreement_models=bundle.get("disagreement_models"),
    )
    pair_df = _augment_inference_features(pair_df, bundle)
    for tier, model in bundle["router_models"].items():
        pair_df[f"pred__tier_{tier}"] = model.predict_proba(
            pair_df[bundle["router_feature_cols"]].to_numpy()
        )[:, 1]
    pair_df["pred__candidate"] = bundle["candidate_model"].predict_proba(
        pair_df[bundle["candidate_feature_cols"]].to_numpy()
    )[:, 1]
    pair_df = _split_column(pair_df, bundle)
    return pair_df, bundle


def _best_rows_by_tier(frame: pd.DataFrame, tier: str) -> pd.DataFrame:
    if tier in {"A", "B"}:
        eligible = frame[frame["router__target_valid_forces"] >= 1.0].copy()
    else:
        eligible = frame[frame["router__target_valid_energy"] >= 1.0].copy()
    idx = eligible.groupby("structure_id")[f"pred__tier_{tier}"].idxmax()
    best = eligible.loc[idx].copy()
    return best[
        [
            "structure_id",
            "immutable_id",
            "model_name",
            f"pred__tier_{tier}",
            f"target__tier_{tier}",
            "target__energy_err_pa",
            "target__force_mean_err",
            "target__force_p90_err",
            "structure__ood_knn_mean",
            "router__energy_ens_std",
            "router__energy_pair_max",
            "router__force_pair_mean",
            "router__force_pair_max",
            "split",
        ]
    ].rename(
        columns={
            "model_name": f"{tier}_model",
            f"pred__tier_{tier}": f"{tier}_prob",
            f"target__tier_{tier}": f"{tier}_hit",
            "target__energy_err_pa": f"{tier}_energy_err_pa",
            "target__force_mean_err": f"{tier}_force_mean_err",
            "target__force_p90_err": f"{tier}_force_p90_err",
            "structure__ood_knn_mean": f"{tier}_ood",
            "router__energy_ens_std": f"{tier}_energy_std",
            "router__energy_pair_max": f"{tier}_energy_pair_max",
            "router__force_pair_mean": f"{tier}_force_pair_mean",
            "router__force_pair_max": f"{tier}_force_pair_max",
            "split": f"{tier}_split",
        }
    )


def build_structure_policy_table(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[["structure_id", "immutable_id", "split"]].drop_duplicates("structure_id").copy()
    best_a = _best_rows_by_tier(frame, "A")
    best_b = _best_rows_by_tier(frame, "B")
    best_c = _best_rows_by_tier(frame, "C")
    merged = base.merge(best_a, on=["structure_id", "immutable_id"], how="left")
    merged = merged.merge(best_b, on=["structure_id", "immutable_id"], how="left")
    merged = merged.merge(best_c, on=["structure_id", "immutable_id"], how="left")
    return merged


def evaluate_policy(policy_table: pd.DataFrame, policy: PolicySpec) -> tuple[pd.DataFrame, dict]:
    rows = []
    for row in policy_table.itertuples(index=False):
        decision = {
            "structure_id": int(row.structure_id),
            "immutable_id": row.immutable_id,
            "split": row.split,
            "status": "reject",
            "chosen_tier": "R",
            "chosen_model": None,
            "confidence": float("nan"),
            "target__tier_hit": 0.0,
            "target__energy_err_pa": float("nan"),
            "target__force_mean_err": float("nan"),
            "target__force_p90_err": float("nan"),
            "ood_score": float("nan"),
            "energy_disagreement": float("nan"),
            "force_disagreement": float("nan"),
        }
        for tier in policy.tier_order:
            threshold = policy.thresholds[tier]
            prob = getattr(row, f"{tier}_prob")
            if pd.notna(prob) and prob >= threshold:
                decision.update(
                    {
                        "status": "accept",
                        "chosen_tier": tier,
                        "chosen_model": getattr(row, f"{tier}_model"),
                        "confidence": float(prob),
                        "target__tier_hit": float(getattr(row, f"{tier}_hit")),
                        "target__energy_err_pa": float(getattr(row, f"{tier}_energy_err_pa")),
                        "target__force_mean_err": float(getattr(row, f"{tier}_force_mean_err")),
                        "target__force_p90_err": float(getattr(row, f"{tier}_force_p90_err")),
                        "ood_score": float(getattr(row, f"{tier}_ood")),
                        "energy_disagreement": float(getattr(row, f"{tier}_energy_std")),
                        "force_disagreement": float(getattr(row, f"{tier}_force_pair_mean")),
                    }
                )
                break
        if decision["status"] == "reject":
            proxy_tier = "B" if pd.notna(getattr(row, "B_prob")) else "C"
            decision.update(
                {
                    "chosen_model": getattr(row, f"{proxy_tier}_model"),
                    "confidence": float(getattr(row, f"{proxy_tier}_prob")),
                    "ood_score": float(getattr(row, f"{proxy_tier}_ood")),
                    "energy_disagreement": float(getattr(row, f"{proxy_tier}_energy_std")),
                    "force_disagreement": float(getattr(row, f"{proxy_tier}_force_pair_mean")),
                }
            )
        rows.append(decision)
    decisions = pd.DataFrame(rows)
    accepted = decisions[decisions["status"] == "accept"].copy()
    summary = {
        "policy": policy.name,
        "description": policy.description,
        "thresholds": policy.thresholds,
        "tier_order": policy.tier_order,
        "num_structures": int(len(decisions)),
        "coverage": float(len(accepted) / len(decisions)) if len(decisions) else 0.0,
        "precision": float(accepted["target__tier_hit"].mean()) if len(accepted) else 0.0,
        "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
        "mean_force_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
        "mean_force_p90_err": float(accepted["target__force_p90_err"].mean()) if len(accepted) else float("nan"),
        "tier_counts": accepted["chosen_tier"].value_counts().to_dict(),
        "model_counts": accepted["chosen_model"].value_counts().to_dict(),
    }
    return decisions, summary


def search_ab_policy(
    policy_table: pd.DataFrame,
    *,
    precision_target: float,
    name: str,
    description: str,
) -> tuple[PolicySpec, dict]:
    best_policy = None
    best_summary = None
    a_grid = np.round(np.arange(0.55, 0.96, 0.02), 2)
    b_grid = np.round(np.arange(0.45, 0.96, 0.02), 2)
    for a_thr in a_grid:
        for b_thr in b_grid:
            policy = PolicySpec(
                name=name,
                tier_order=["A", "B"],
                thresholds={"A": float(a_thr), "B": float(b_thr), "C": 1.01},
                description=description,
            )
            _, summary = evaluate_policy(policy_table, policy)
            if summary["precision"] < precision_target:
                continue
            if best_summary is None or summary["coverage"] > best_summary["coverage"] or (
                np.isclose(summary["coverage"], best_summary["coverage"])
                and summary["precision"] > best_summary["precision"]
            ):
                best_policy = policy
                best_summary = summary
    if best_policy is None:
        fallback = PolicySpec(
            name=name,
            tier_order=["A", "B"],
            thresholds={"A": 0.90, "B": 0.90, "C": 1.01},
            description=description,
        )
        _, best_summary = evaluate_policy(policy_table, fallback)
        best_policy = fallback
    return best_policy, best_summary


def search_c_policy(
    policy_table: pd.DataFrame,
    *,
    precision_target: float,
    name: str,
    description: str,
) -> tuple[PolicySpec, dict]:
    best_policy = None
    best_summary = None
    c_grid = np.round(np.arange(0.05, 0.98, 0.02), 2)
    for c_thr in c_grid:
        policy = PolicySpec(
            name=name,
            tier_order=["C"],
            thresholds={"A": 1.01, "B": 1.01, "C": float(c_thr)},
            description=description,
        )
        _, summary = evaluate_policy(policy_table, policy)
        if summary["precision"] < precision_target:
            continue
        if best_summary is None or summary["coverage"] > best_summary["coverage"] or (
            np.isclose(summary["coverage"], best_summary["coverage"])
            and summary["precision"] > best_summary["precision"]
        ):
            best_policy = policy
            best_summary = summary
    if best_policy is None:
        fallback = PolicySpec(
            name=name,
            tier_order=["C"],
            thresholds={"A": 1.01, "B": 1.01, "C": 0.90},
            description=description,
        )
        _, best_summary = evaluate_policy(policy_table, fallback)
        best_policy = fallback
    return best_policy, best_summary


def tag_rejects(decisions: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    reject_mask = out["status"] == "reject"
    if reject_mask.sum() == 0:
        out["reject_reason"] = ""
        out["dft_priority_score"] = 0.0
        return out

    reject = out.loc[reject_mask].copy()
    ood_q90 = reject["ood_score"].quantile(0.90)
    energy_q90 = reject["energy_disagreement"].quantile(0.90)
    force_q90 = reject["force_disagreement"].quantile(0.90)
    conf_q25 = reject["confidence"].quantile(0.25)

    ood_rank = reject["ood_score"].rank(pct=True, method="average")
    energy_rank = reject["energy_disagreement"].rank(pct=True, method="average")
    force_rank = reject["force_disagreement"].rank(pct=True, method="average")
    confidence_rank = 1.0 - reject["confidence"].rank(pct=True, method="average")
    score_map = {
        idx: float(ood_rank.loc[idx] + energy_rank.loc[idx] + force_rank.loc[idx] + confidence_rank.loc[idx])
        for idx in reject.index
    }

    reasons = []
    scores = []
    for idx, row in zip(out.index, out.itertuples(index=False), strict=True):
        if row.status != "reject":
            reasons.append("")
            scores.append(0.0)
            continue
        if row.ood_score >= ood_q90:
            reason = "high_ood"
        elif row.force_disagreement >= force_q90 or row.energy_disagreement >= energy_q90:
            reason = "high_disagreement"
        elif row.confidence >= conf_q25:
            reason = "frontier_uncertainty"
        else:
            reason = "low_confidence"
        reasons.append(reason)
        scores.append(score_map[idx])
    out["reject_reason"] = reasons
    out["dft_priority_score"] = scores
    return out


def _write_report(
    path: Path,
    *,
    base_summary: dict,
    policy_summaries: list[dict],
) -> None:
    lines = []
    lines.append("# Selective Router Analysis Report")
    lines.append("")
    lines.append("## Base Router")
    lines.append("")
    lines.append(f"- coverage: {base_summary['coverage']:.4f}")
    lines.append(f"- precision: {base_summary['precision']:.4f}")
    lines.append(f"- mean energy error: {base_summary['mean_energy_err_pa']:.4f} eV/atom")
    lines.append(f"- mean force error: {base_summary['mean_force_err']:.4f} eV/A")
    lines.append(f"- mean force p90 error: {base_summary['mean_force_p90_err']:.4f} eV/A")
    lines.append("")
    lines.append("## Recommended Policies")
    lines.append("")
    for summary in policy_summaries:
        lines.append(f"### {summary['policy']}")
        lines.append("")
        lines.append(f"- description: {summary['description']}")
        lines.append(f"- tier order: {summary['tier_order']}")
        lines.append(f"- thresholds: {summary['thresholds']}")
        lines.append(f"- coverage: {summary['coverage']:.4f}")
        lines.append(f"- precision: {summary['precision']:.4f}")
        lines.append(f"- mean energy error: {summary['mean_energy_err_pa']:.4f} eV/atom")
        lines.append(f"- mean force error: {summary['mean_force_err']:.4f} eV/A")
        lines.append(f"- mean force p90 error: {summary['mean_force_p90_err']:.4f} eV/A")
        lines.append(f"- tier counts: {summary['tier_counts']}")
        lines.append("")
    path.write_text("\n".join(lines))


def run_policy_analysis(
    root: Path,
    bundle_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_df, bundle = build_scored_pair_frame(root, bundle_path)
    policy_table = build_structure_policy_table(pair_df)

    test_table = policy_table[policy_table["split"] == "test"].copy()
    full_table = policy_table.copy()

    base_policy = PolicySpec(
        name="bundle_default",
        tier_order=TIER_ORDER,
        thresholds=bundle["thresholds"],
        description="Original validation-tuned hierarchical policy from training.",
    )
    _, base_summary = evaluate_policy(test_table, base_policy)

    policies: list[PolicySpec] = []
    policy_summaries: list[dict] = []

    ab_strict, _ = search_ab_policy(
        test_table,
        precision_target=0.90,
        name="ab_strict",
        description="High-confidence energy+forces pool for pseudo-label pretraining.",
    )
    ab_balanced, _ = search_ab_policy(
        test_table,
        precision_target=0.85,
        name="ab_balanced",
        description="Larger energy+forces pool with slightly relaxed precision.",
    )
    c_strict, _ = search_c_policy(
        test_table,
        precision_target=0.93,
        name="c_energy_only",
        description="High-confidence energy-only expansion pool.",
    )
    policies.extend([ab_strict, ab_balanced, c_strict])

    export_rows = []
    policy_json = {}
    for policy in policies:
        test_decisions, test_summary = evaluate_policy(test_table, policy)
        test_decisions = tag_rejects(test_decisions)
        test_decisions.to_csv(output_dir / f"test_{policy.name}_decisions.csv", index=False)

        full_decisions, full_summary = evaluate_policy(full_table, policy)
        full_decisions = tag_rejects(full_decisions)
        full_decisions.to_csv(output_dir / f"all_{policy.name}_decisions.csv", index=False)

        rejects = full_decisions[full_decisions["status"] == "reject"].copy()
        rejects = rejects.sort_values("dft_priority_score", ascending=False)
        rejects.head(1000).to_csv(output_dir / f"all_{policy.name}_dft_candidates_top1000.csv", index=False)

        summary = {
            "policy": policy.name,
            "description": policy.description,
            "tier_order": policy.tier_order,
            "thresholds": policy.thresholds,
            "test": test_summary,
            "all_labeled": full_summary,
        }
        policy_summaries.append(test_summary)
        export_rows.append(summary)
        policy_json[policy.name] = {
            "description": policy.description,
            "tier_order": policy.tier_order,
            "thresholds": policy.thresholds,
        }

    (output_dir / "policy_summary.json").write_text(json.dumps(export_rows, indent=2))
    (output_dir / "policy_config.json").write_text(json.dumps(policy_json, indent=2))
    _write_report(output_dir / "analysis_report.md", base_summary=base_summary, policy_summaries=policy_summaries)
    return {
        "base_summary": base_summary,
        "policies": export_rows,
    }
