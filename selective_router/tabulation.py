from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .constants import DEFAULT_TIERS


BACKEND_INFO = {
    "random_forest": {
        "label": "RandomForest",
        "train_dir": Path("artifacts/selective_router_full"),
        "analysis_dir": Path("artifacts/selective_router_analysis"),
    },
    "xgboost": {
        "label": "XGBoost",
        "train_dir": Path("artifacts/xgb_full_env1"),
        "analysis_dir": Path("artifacts/xgb_analysis_env1"),
    },
    "lightgbm": {
        "label": "LightGBM",
        "train_dir": Path("artifacts/lgbm_full_env1_rerun"),
        "analysis_dir": Path("artifacts/lgbm_analysis_env1_rerun"),
    },
}


def _policy_rows(root: Path) -> pd.DataFrame:
    rows = []
    for backend, info in BACKEND_INFO.items():
        train_summary = json.loads((root / info["train_dir"] / "summary.json").read_text())
        policies = json.loads((root / info["analysis_dir"] / "policy_summary.json").read_text())
        for item in policies:
            test = item["test"]
            rows.append(
                {
                    "backend": info["label"],
                    "policy": item["policy"],
                    "threshold_A": item["thresholds"]["A"],
                    "threshold_B": item["thresholds"]["B"],
                    "threshold_C": item["thresholds"]["C"],
                    "coverage": round(test["coverage"], 4),
                    "precision": round(test["precision"], 4),
                    "energy_err_pa": round(test["mean_energy_err_pa"], 4),
                    "force_err": round(test["mean_force_err"], 4),
                    "force_p90_err": round(test["mean_force_p90_err"], 4),
                    "recall_at_1": round(train_summary["candidate_metrics"]["test"]["recall@1"], 4),
                    "recall_at_3": round(train_summary["candidate_metrics"]["test"]["recall@3"], 4),
                }
            )
    return pd.DataFrame(rows)


def _tier_definition_rows() -> pd.DataFrame:
    rows = []
    for name, spec in DEFAULT_TIERS.items():
        rows.append(
            {
                "tier": name,
                "energy_max_eV_per_atom": spec.energy_pa_max,
                "force_mean_max_eV_per_A": spec.force_mean_max,
                "force_p90_max_eV_per_A": spec.force_p90_max,
                "meaning": {
                    "A": "high-quality energy+forces",
                    "B": "relaxed energy+forces",
                    "C": "energy-only",
                }[name],
            }
        )
    rows.append(
        {
            "tier": "R",
            "energy_max_eV_per_atom": None,
            "force_mean_max_eV_per_A": None,
            "force_p90_max_eV_per_A": None,
            "meaning": "reject / incremental DFT",
        }
    )
    return pd.DataFrame(rows)


def _teacher_usage_rows(root: Path) -> pd.DataFrame:
    decisions = pd.read_csv(root / "artifacts/xgb_analysis_env1/all_ab_strict_decisions.csv")
    accepted = decisions[decisions["status"] == "accept"].copy()
    counts = accepted["chosen_model"].value_counts()
    shares = counts / counts.sum()

    rows = []
    for rank, (model_name, count) in enumerate(counts.items(), start=1):
        share = float(shares.loc[model_name])
        if share >= 0.50:
            role = "main teacher"
        elif share >= 0.15:
            role = "secondary teacher"
        elif share >= 0.03:
            role = "supplementary teacher"
        else:
            role = "rarely selected"
        rows.append(
            {
                "rank": rank,
                "teacher_model": model_name,
                "selected_count": int(count),
                "selected_share": round(share, 4),
                "recommended_role": role,
            }
        )
    return pd.DataFrame(rows)


def _teacher_cluster_profile_rows(root: Path) -> pd.DataFrame:
    cluster = pd.read_csv(root / "artifacts/xgb_full_env1/cluster_model_profile.csv")
    rows = []
    for model_name, frame in cluster.groupby("model_name"):
        weight = frame["count"]
        rows.append(
            {
                "teacher_model": model_name,
                "tier_A_rate": round(float((frame["tier_A_rate"] * weight).sum() / weight.sum()), 4),
                "tier_B_rate": round(float((frame["tier_B_rate"] * weight).sum() / weight.sum()), 4),
                "tier_C_rate": round(float((frame["tier_C_rate"] * weight).sum() / weight.sum()), 4),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["tier_B_rate", "tier_A_rate", "tier_C_rate"], ascending=[False, False, False]
    )
    return out.reset_index(drop=True)


def _write_markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", "", frame.to_markdown(index=False)]
    path.write_text("\n".join(lines))


def _draw_table(frame: pd.DataFrame, path: Path, title: str, figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=frame.astype(str).values,
        colLabels=frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title(title, pad=16, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_summary_tables(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_table = _policy_rows(root)
    tier_table = _tier_definition_rows()
    teacher_usage = _teacher_usage_rows(root)
    teacher_profile = _teacher_cluster_profile_rows(root)

    backend_csv = output_dir / "backend_policy_table.csv"
    tier_csv = output_dir / "tier_definition_table.csv"
    usage_csv = output_dir / "teacher_usage_xgb_ab_strict.csv"
    profile_csv = output_dir / "teacher_quality_profile.csv"

    backend_table.to_csv(backend_csv, index=False)
    tier_table.to_csv(tier_csv, index=False)
    teacher_usage.to_csv(usage_csv, index=False)
    teacher_profile.to_csv(profile_csv, index=False)

    _write_markdown_table(backend_table, output_dir / "backend_policy_table.md", "Backend Policy Table")
    _write_markdown_table(tier_table, output_dir / "tier_definition_table.md", "Tier Definition Table")
    _write_markdown_table(teacher_usage, output_dir / "teacher_usage_xgb_ab_strict.md", "Teacher Usage Table")
    _write_markdown_table(
        teacher_profile, output_dir / "teacher_quality_profile.md", "Teacher Quality Profile Table"
    )

    _draw_table(
        backend_table,
        output_dir / "backend_policy_table.png",
        "Backend Performance / Threshold Summary",
        figsize=(16, 4.8),
    )
    _draw_table(
        tier_table,
        output_dir / "tier_definition_table.png",
        "A/B/C/R Definition Summary",
        figsize=(12, 2.8),
    )
    _draw_table(
        teacher_usage,
        output_dir / "teacher_usage_xgb_ab_strict.png",
        "Teacher Usage Under XGBoost + ab_strict",
        figsize=(12, 4.0),
    )
    _draw_table(
        teacher_profile,
        output_dir / "teacher_quality_profile.png",
        "Teacher Quality Profile (Weighted Cluster Rates)",
        figsize=(12, 4.2),
    )

    return {
        "output_dir": str(output_dir),
        "tables": [
            str(backend_csv),
            str(tier_csv),
            str(usage_csv),
            str(profile_csv),
        ],
        "markdown": [
            str(output_dir / "backend_policy_table.md"),
            str(output_dir / "tier_definition_table.md"),
            str(output_dir / "teacher_usage_xgb_ab_strict.md"),
            str(output_dir / "teacher_quality_profile.md"),
        ],
        "images": [
            str(output_dir / "backend_policy_table.png"),
            str(output_dir / "tier_definition_table.png"),
            str(output_dir / "teacher_usage_xgb_ab_strict.png"),
            str(output_dir / "teacher_quality_profile.png"),
        ],
    }
