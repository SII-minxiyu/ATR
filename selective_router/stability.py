from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import run_policy_analysis
from .pipeline import train_experiment


def _policy_by_name(policies: list[dict], name: str) -> dict:
    for policy in policies:
        if policy["policy"] == name:
            return policy
    raise KeyError(f"Policy not found: {name}")


def _seed_output_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / f"seed_{seed}"


def _teacher_rows(seed: int, model_counts: dict[str, int]) -> list[dict]:
    total = sum(model_counts.values())
    ordered = sorted(model_counts.items(), key=lambda item: (-item[1], item[0]))
    rows = []
    for rank, (model_name, count) in enumerate(ordered, start=1):
        rows.append(
            {
                "seed": seed,
                "rank": rank,
                "teacher_model": model_name,
                "count": int(count),
                "share": float(count / total) if total else 0.0,
            }
        )
    return rows


def _bootstrap_metrics(decisions: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(decisions)
    rows = []
    for bootstrap_id in range(n_bootstrap):
        sampled_ids = rng.integers(0, n, size=n)
        sample = decisions.iloc[sampled_ids].copy()
        accepted = sample[sample["status"] == "accept"].copy()
        row = {
            "bootstrap_id": bootstrap_id,
            "coverage": float(len(accepted) / n) if n else 0.0,
            "precision": float(accepted["target__tier_hit"].mean()) if len(accepted) else 0.0,
            "mean_energy_err_pa": float(accepted["target__energy_err_pa"].mean()) if len(accepted) else float("nan"),
            "mean_force_err": float(accepted["target__force_mean_err"].mean()) if len(accepted) else float("nan"),
            "mean_force_p90_err": float(accepted["target__force_p90_err"].mean()) if len(accepted) else float("nan"),
        }
        if len(accepted):
            share = accepted["chosen_model"].value_counts(normalize=True)
            for model_name, value in share.items():
                row[f"share__{model_name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _interval_summary(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "coverage",
        "precision",
        "mean_energy_err_pa",
        "mean_force_err",
        "mean_force_p90_err",
    ]
    rows = []
    for col in metric_cols:
        series = bootstrap_df[col].dropna()
        rows.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "p05": float(series.quantile(0.05)),
                "p50": float(series.quantile(0.50)),
                "p95": float(series.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _teacher_stability_summary(teacher_df: pd.DataFrame) -> pd.DataFrame:
    seed_count = teacher_df["seed"].nunique()
    rows = []
    for model_name, frame in teacher_df.groupby("teacher_model"):
        rows.append(
            {
                "teacher_model": model_name,
                "rank_mean": float(frame["rank"].mean()),
                "rank_std": float(frame["rank"].std(ddof=0)),
                "share_mean": float(frame["share"].mean()),
                "share_std": float(frame["share"].std(ddof=0)),
                "top1_freq": float((frame["rank"] == 1).sum() / seed_count),
                "top3_freq": float((frame["rank"] <= 3).sum() / seed_count),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank_mean", "share_mean"], ascending=[True, False])


def _write_report(
    path: Path,
    *,
    backend: str,
    policy_name: str,
    seeds: list[int],
    metrics_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Stability Report",
        "",
        f"- backend: `{backend}`",
        f"- policy: `{policy_name}`",
        f"- seeds: `{seeds}`",
        "",
        "## Multi-Seed Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Teacher Ranking Stability",
        "",
        teacher_df.to_markdown(index=False),
        "",
        "## Bootstrap Intervals",
        "",
        bootstrap_summary.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines))


def run_stability_analysis(
    root: Path,
    output_dir: Path,
    *,
    backend: str = "xgboost",
    seeds: list[int] | None = None,
    policy_name: str = "ab_strict",
    n_bootstrap: int = 500,
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42, 84, 168]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    teacher_rows = []
    decisions_for_bootstrap: pd.DataFrame | None = None

    for seed in seeds:
        seed_dir = _seed_output_dir(output_dir, seed)
        train_dir = seed_dir / "train"
        analysis_dir = seed_dir / "analysis"
        train_summary = train_experiment(
            root=root,
            output_dir=train_dir,
            seed=seed,
            backend=backend,
        )
        analysis_summary = run_policy_analysis(
            root=root,
            bundle_path=train_dir / "router_bundle.joblib",
            output_dir=analysis_dir,
        )
        policy = _policy_by_name(analysis_summary["policies"], policy_name)
        test = policy["test"]
        metric_rows.append(
            {
                "seed": seed,
                "threshold_A": policy["thresholds"]["A"],
                "threshold_B": policy["thresholds"]["B"],
                "threshold_C": policy["thresholds"]["C"],
                "coverage": test["coverage"],
                "precision": test["precision"],
                "mean_energy_err_pa": test["mean_energy_err_pa"],
                "mean_force_err": test["mean_force_err"],
                "mean_force_p90_err": test["mean_force_p90_err"],
                "candidate_recall1": train_summary["candidate_metrics"]["test"]["recall@1"],
                "candidate_recall3": train_summary["candidate_metrics"]["test"]["recall@3"],
            }
        )
        teacher_rows.extend(_teacher_rows(seed, policy["all_labeled"]["model_counts"]))
        if decisions_for_bootstrap is None:
            decisions_for_bootstrap = pd.read_csv(analysis_dir / f"test_{policy_name}_decisions.csv")

    metrics_df = pd.DataFrame(metric_rows).sort_values("seed")
    metrics_df.to_csv(output_dir / "multi_seed_metrics.csv", index=False)
    teacher_rank_df = pd.DataFrame(teacher_rows).sort_values(["seed", "rank"])
    teacher_rank_df.to_csv(output_dir / "teacher_rankings_by_seed.csv", index=False)

    metric_summary = metrics_df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"]).T.reset_index()
    metric_summary = metric_summary.rename(columns={"index": "metric"})
    metric_summary.to_csv(output_dir / "multi_seed_metric_summary.csv", index=False)

    teacher_summary = _teacher_stability_summary(teacher_rank_df)
    teacher_summary.to_csv(output_dir / "teacher_stability_summary.csv", index=False)

    if decisions_for_bootstrap is None:
        raise RuntimeError("Bootstrap decisions were not generated.")
    bootstrap_df = _bootstrap_metrics(decisions_for_bootstrap, n_bootstrap=n_bootstrap, seed=seeds[0])
    bootstrap_df.to_csv(output_dir / "bootstrap_samples.csv", index=False)
    bootstrap_summary = _interval_summary(bootstrap_df)
    bootstrap_summary.to_csv(output_dir / "bootstrap_metric_summary.csv", index=False)

    _write_report(
        output_dir / "stability_report.md",
        backend=backend,
        policy_name=policy_name,
        seeds=seeds,
        metrics_df=metrics_df,
        teacher_df=teacher_summary,
        bootstrap_summary=bootstrap_summary,
    )

    return {
        "backend": backend,
        "policy": policy_name,
        "seeds": seeds,
        "n_bootstrap": n_bootstrap,
        "output_dir": str(output_dir),
        "metric_summary": metric_summary.to_dict(orient="records"),
        "teacher_stability_summary": teacher_summary.to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
    }
