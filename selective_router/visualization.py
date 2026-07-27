from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BACKEND_INFO = {
    "random_forest": {
        "label": "RandomForest",
        "color": "#4C78A8",
        "train_dir": Path("artifacts/selective_router_full"),
        "analysis_dir": Path("artifacts/selective_router_analysis"),
    },
    "xgboost": {
        "label": "XGBoost",
        "color": "#F58518",
        "train_dir": Path("artifacts/xgb_full_env1"),
        "analysis_dir": Path("artifacts/xgb_analysis_env1"),
    },
    "lightgbm": {
        "label": "LightGBM",
        "color": "#54A24B",
        "train_dir": Path("artifacts/lgbm_full_env1_rerun"),
        "analysis_dir": Path("artifacts/lgbm_analysis_env1_rerun"),
    },
}

TIER_COLORS = {
    "A": "#2E8B57",
    "B": "#4C78A8",
    "C": "#F58518",
    "R": "#9D9D9D",
}

POLICY_MARKERS = {
    "ab_strict": "o",
    "ab_balanced": "s",
    "c_energy_only": "^",
}


def _load_policy_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _load_backend_tables(root: Path) -> dict[str, dict]:
    tables: dict[str, dict] = {}
    for backend, info in BACKEND_INFO.items():
        train_dir = root / info["train_dir"]
        analysis_dir = root / info["analysis_dir"]
        tables[backend] = {
            "label": info["label"],
            "color": info["color"],
            "risk_curve": pd.read_csv(train_dir / "risk_coverage_curve.csv"),
            "train_summary": json.loads((train_dir / "summary.json").read_text()),
            "policy_summary": _load_policy_summary(analysis_dir / "policy_summary.json"),
        }
    return tables


def _counts_from_decisions(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    counts = frame["chosen_tier"].fillna("R").value_counts().to_dict()
    return {tier: int(counts.get(tier, 0)) for tier in ["A", "B", "C", "R"]}


def plot_risk_coverage(root: Path, output_dir: Path) -> Path:
    tables = _load_backend_tables(root)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("precision", "Coverage", "Precision", "Coverage vs Precision"),
        ("mean_force_err", "Coverage", "Mean force error (eV/A)", "Coverage vs Force Error"),
        ("mean_energy_err_pa", "Coverage", "Mean energy error (eV/atom)", "Coverage vs Energy Error"),
    ]

    for ax, (metric, xlabel, ylabel, title) in zip(axes, metrics):
        for backend, payload in tables.items():
            curve = payload["risk_curve"]
            ax.plot(
                curve["coverage"],
                curve[metric],
                label=payload["label"],
                color=payload["color"],
                linewidth=2,
            )
            for policy in payload["policy_summary"]:
                point = policy["test"]
                ax.scatter(
                    point["coverage"],
                    point[metric],
                    color=payload["color"],
                    marker=POLICY_MARKERS[policy["policy"]],
                    s=70,
                    edgecolors="black",
                    linewidths=0.6,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Backend Risk-Coverage Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    out = output_dir / "risk_coverage_backends.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_abcr_distribution(root: Path, output_dir: Path, split: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    policies = ["ab_strict", "ab_balanced", "c_energy_only"]

    for ax, policy_name in zip(axes, policies):
        backend_labels: list[str] = []
        bottoms = [0.0, 0.0, 0.0]
        tier_to_values: dict[str, list[float]] = {tier: [] for tier in ["A", "B", "C", "R"]}

        for backend, info in BACKEND_INFO.items():
            analysis_dir = root / info["analysis_dir"]
            decision_name = f"{split}_{policy_name}_decisions.csv"
            counts = _counts_from_decisions(analysis_dir / decision_name)
            total = sum(counts.values())
            backend_labels.append(info["label"])
            for tier in tier_to_values:
                tier_to_values[tier].append(counts[tier] / total if total else 0.0)

        for tier in ["A", "B", "C", "R"]:
            values = tier_to_values[tier]
            ax.bar(
                backend_labels,
                values,
                bottom=bottoms,
                color=TIER_COLORS[tier],
                label=tier,
                width=0.62,
            )
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_ylim(0, 1.0)
        ax.set_title(policy_name)
        ax.set_ylabel("Fraction of structures")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"ABCR Distribution ({split})", y=1.03, fontsize=14)
    fig.tight_layout()
    out = output_dir / f"abcr_distribution_{split}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_policy_scatter(root: Path, output_dir: Path) -> Path:
    tables = _load_backend_tables(root)
    fig, ax = plt.subplots(figsize=(8, 6))

    for backend, payload in tables.items():
        for policy in payload["policy_summary"]:
            point = policy["test"]
            ax.scatter(
                point["coverage"],
                point["precision"],
                color=payload["color"],
                marker=POLICY_MARKERS[policy["policy"]],
                s=110,
                edgecolors="black",
                linewidths=0.7,
            )
            ax.annotate(
                f"{payload['label']}\n{policy['policy']}",
                (point["coverage"], point["precision"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision")
    ax.set_title("Policy-Level Coverage-Precision Tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = output_dir / "policy_tradeoff_scatter.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_visual_report(root: Path, output_dir: Path, generated: list[Path]) -> Path:
    lines = [
        "# Visualization Report",
        "",
        "## Figures",
        "",
        "- `risk_coverage_backends.png`: three backends on the same risk-coverage curves.",
        "- `abcr_distribution_test.png`: held-out test split ABCR fractions.",
        "- `abcr_distribution_all.png`: all 18k labeled samples under each policy.",
        "- `policy_tradeoff_scatter.png`: policy-level coverage vs precision comparison.",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in generated)
    report = output_dir / "visualization_report.md"
    report.write_text("\n".join(lines))
    return report


def generate_all_visualizations(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_risk_coverage(root, output_dir),
        plot_abcr_distribution(root, output_dir, "test"),
        plot_abcr_distribution(root, output_dir, "all"),
        plot_policy_scatter(root, output_dir),
    ]
    report = write_visual_report(root, output_dir, generated)
    return {
        "output_dir": str(output_dir),
        "figures": [str(path) for path in generated],
        "report": str(report),
    }
