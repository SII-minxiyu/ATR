from __future__ import annotations

import json
from pathlib import Path

from .analysis import run_policy_analysis
from .pipeline import train_experiment


BACKENDS = ["random_forest", "xgboost", "lightgbm"]


def run_backend_comparison(
    root: Path,
    output_dir: Path,
    seed: int = 42,
    max_records: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    for backend in BACKENDS:
        backend_dir = output_dir / backend
        train_summary = train_experiment(
            root=root,
            output_dir=backend_dir / "train",
            seed=seed,
            backend=backend,
            max_records=max_records,
        )
        analysis_summary = run_policy_analysis(
            root=root,
            bundle_path=backend_dir / "train" / "router_bundle.joblib",
            output_dir=backend_dir / "analysis",
        )
        comparison.append(
            {
                "backend": backend,
                "train_summary": train_summary,
                "analysis_summary": analysis_summary,
            }
        )

    (output_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2))

    lines = []
    lines.append("# Backend Comparison Report")
    lines.append("")
    for item in comparison:
        backend = item["backend"]
        train_summary = item["train_summary"]
        base = item["analysis_summary"]["base_summary"]
        lines.append(f"## {backend}")
        lines.append("")
        lines.append(f"- base coverage: {base['coverage']:.4f}")
        lines.append(f"- base precision: {base['precision']:.4f}")
        lines.append(f"- base mean energy error: {base['mean_energy_err_pa']:.4f} eV/atom")
        lines.append(f"- base mean force error: {base['mean_force_err']:.4f} eV/A")
        for policy in item["analysis_summary"]["policies"]:
            test = policy["test"]
            lines.append(f"- {policy['policy']}: coverage={test['coverage']:.4f}, precision={test['precision']:.4f}, "
                         f"E={test['mean_energy_err_pa']:.4f}, F={test['mean_force_err']:.4f}")
        lines.append("")
    (output_dir / "comparison_report.md").write_text("\n".join(lines))
    return {"backends": comparison}
