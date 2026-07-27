from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_router.ar_maxforce import (
    LIGHT_POSTPROCESS_CONFIDENCE_LT,
    LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
    run_ar_postprocess_analysis,
    route_ar_structures,
    run_ar_maxforce_cross_validation,
    run_ar_maxforce_experiment,
)
from selective_router.analysis import run_policy_analysis
from selective_router.compare import run_backend_comparison
from selective_router.extxyz_candidate import rank_extxyz_candidates
from selective_router.pipeline import rank_models_for_structure, route_structures, train_experiment
from selective_router.stability import run_stability_analysis
from selective_router.tabulation import generate_summary_tables
from selective_router.visualization import generate_all_visualizations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective pseudo-label router workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the selective router on the 18k labeled set.")
    train_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/selective_router"),
        help="Directory to store models and reports.",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--backend",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Meta-model backend for candidate ranking and routing.",
    )
    train_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for debugging on a smaller labeled subset.",
    )
    train_parser.add_argument(
        "--disagreement-mode",
        type=str,
        default="all",
        choices=["all", "top3", "top5"],
        help="Which teacher subset to use when computing multi-teacher disagreement features.",
    )

    rank_parser = subparsers.add_parser("rank", help="Rank candidate teachers for a structure before inference.")
    rank_parser.add_argument("--bundle", type=Path, required=True, help="Path to router_bundle.joblib")
    rank_parser.add_argument("--structure-db", type=Path, required=True, help="ASE db with structures.")
    rank_parser.add_argument("--structure-id", type=int, required=True, help="Structure id inside the db.")
    rank_parser.add_argument("--top-k", type=int, default=3)

    rank_extxyz_parser = subparsers.add_parser(
        "rank-extxyz",
        help="Rank candidate teachers for structures stored in an ExtXYZ file.",
    )
    rank_extxyz_parser.add_argument("--bundle", type=Path, required=True, help="Path to router_bundle.joblib")
    rank_extxyz_parser.add_argument("--xyz", type=Path, required=True, help="ExtXYZ structure file.")
    rank_extxyz_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with 18k dbs.")
    rank_extxyz_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/extxyz_candidate_ranking"),
        help="Directory to store candidate ranking outputs.",
    )
    rank_extxyz_parser.add_argument("--top-k", type=int, default=3)
    rank_extxyz_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for smoke testing on the first N structures.",
    )
    rank_extxyz_parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Optionally rank every Nth structure for a cheaper stratified pass over the full file.",
    )
    rank_extxyz_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset used together with --every-n when sampling through the file.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Search stricter pseudo-label policies and export pool reports."
    )
    analyze_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    analyze_parser.add_argument("--bundle", type=Path, required=True, help="Path to router_bundle.joblib")
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/selective_router_analysis"),
        help="Directory to store policy analysis outputs.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Run full RandomForest vs XGBoost vs LightGBM comparison."
    )
    compare_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backend_comparison"),
        help="Directory to store backend comparison outputs.",
    )
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for debugging backend comparison on a smaller subset.",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate ABCR and risk-coverage visualizations.")
    plot_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with saved artifacts.")
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/router_visualizations"),
        help="Directory to store generated figures.",
    )

    table_parser = subparsers.add_parser("tables", help="Generate summary tables for backends and teachers.")
    table_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with saved artifacts.")
    table_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/router_tables"),
        help="Directory to store generated table files.",
    )

    stability_parser = subparsers.add_parser(
        "stability", help="Run multi-seed and bootstrap stability analysis for one backend."
    )
    stability_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    stability_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/router_stability"),
        help="Directory to store stability outputs.",
    )
    stability_parser.add_argument(
        "--backend",
        type=str,
        default="xgboost",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Backend to stress-test.",
    )
    stability_parser.add_argument(
        "--policy",
        type=str,
        default="ab_strict",
        choices=["ab_strict", "ab_balanced", "c_energy_only"],
        help="Policy to summarize.",
    )
    stability_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 21, 42, 84, 168],
        help="Random seeds for repeated train/val/test regrouping.",
    )
    stability_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap resamples on one held-out test run.",
    )

    route_parser = subparsers.add_parser("route", help="Select or reject after model predictions exist.")
    route_parser.add_argument("--bundle", type=Path, required=True, help="Path to router_bundle.joblib")
    route_parser.add_argument("--structure-db", type=Path, required=True, help="ASE db with structures.")
    route_parser.add_argument(
        "--model-db",
        action="append",
        type=Path,
        default=[],
        help="Prediction db paths. Repeat this flag for each available model db.",
    )
    route_parser.add_argument(
        "--structure-id",
        action="append",
        type=int,
        default=None,
        help="Optional structure id filter. Repeat to score multiple structures.",
    )
    route_parser.add_argument("--output-csv", type=Path, default=None)
    route_parser.add_argument("--threshold-a", type=float, default=None)
    route_parser.add_argument("--threshold-b", type=float, default=None)
    route_parser.add_argument("--threshold-c", type=float, default=None)

    ar_parser = subparsers.add_parser(
        "ar-maxforce",
        help="Run a simplified A/R experiment using energy error and max-force error only.",
    )
    ar_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    ar_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ar_only_maxforce_experiment/results"),
        help="Directory to store experiment outputs.",
    )
    ar_parser.add_argument("--seed", type=int, default=42)
    ar_parser.add_argument(
        "--backend",
        type=str,
        default="xgboost",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Backend to use for the A/R router.",
    )
    ar_parser.add_argument(
        "--disagreement-mode",
        type=str,
        default="all",
        choices=["all", "top3", "top5"],
        help="Which teacher subset to use when computing multi-teacher disagreement features.",
    )
    ar_parser.add_argument(
        "--precision-target",
        type=float,
        default=0.90,
        help="Validation precision target used when choosing the final acceptance threshold.",
    )

    ar_cv_parser = subparsers.add_parser(
        "ar-maxforce-cv",
        help="Run grouped 5-fold cross-validation for the simplified A/R experiment.",
    )
    ar_cv_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    ar_cv_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ar_only_maxforce_experiment/cv_results"),
        help="Directory to store cross-validation outputs.",
    )
    ar_cv_parser.add_argument("--seed", type=int, default=42)
    ar_cv_parser.add_argument(
        "--backend",
        type=str,
        default="xgboost",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Backend to use for the A/R router.",
    )
    ar_cv_parser.add_argument(
        "--disagreement-mode",
        type=str,
        default="all",
        choices=["all", "top3", "top5"],
        help="Which teacher subset to use when computing multi-teacher disagreement features.",
    )
    ar_cv_parser.add_argument(
        "--precision-target",
        type=float,
        default=0.90,
        help="Calibration precision target used when choosing the final threshold inside each fold.",
    )
    ar_cv_parser.add_argument("--n-splits", type=int, default=5)


    ar_route_parser = subparsers.add_parser(
        "ar-route",
        help="Select or reject structures using a trained A/R max-force bundle after teacher predictions exist.",
    )
    ar_route_parser.add_argument("--bundle", type=Path, required=True, help="Path to A/R router_bundle.joblib")
    ar_route_parser.add_argument("--structure-db", type=Path, required=True, help="ASE db with structures.")
    ar_route_parser.add_argument(
        "--model-db",
        action="append",
        type=Path,
        default=[],
        help="Prediction db paths. Repeat this flag for each available model db.",
    )
    ar_route_parser.add_argument(
        "--structure-id",
        action="append",
        type=int,
        default=None,
        help="Optional structure id filter. Repeat to score multiple structures.",
    )
    ar_route_parser.add_argument("--output-csv", type=Path, default=None)
    ar_route_parser.add_argument("--threshold", type=float, default=None)
    ar_route_parser.add_argument(
        "--postprocess-light",
        action="store_true",
        help="Apply the lightweight teacher-only postprocess reject rule after routing.",
    )
    ar_route_parser.add_argument(
        "--postprocess-confidence-lt",
        type=float,
        default=LIGHT_POSTPROCESS_CONFIDENCE_LT,
        help="Reject accepted samples when confidence is below this threshold and force deviation is high.",
    )
    ar_route_parser.add_argument(
        "--postprocess-force-dev-max-gt",
        type=float,
        default=LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
        help="Reject accepted samples when router__target_force_dev_max exceeds this threshold.",
    )

    ar_post_parser = subparsers.add_parser(
        "ar-postprocess",
        help="Apply the lightweight teacher-only postprocess reject rule to an existing A/R experiment.",
    )
    ar_post_parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root with ASE databases.")
    ar_post_parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Existing A/R experiment directory containing all_structure_decisions.csv and summary.json.",
    )
    ar_post_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ar_only_maxforce_postprocess_experiment/results"),
        help="Directory to store postprocessed outputs.",
    )
    ar_post_parser.add_argument(
        "--confidence-lt",
        type=float,
        default=LIGHT_POSTPROCESS_CONFIDENCE_LT,
        help="Postprocess confidence threshold.",
    )
    ar_post_parser.add_argument(
        "--force-dev-max-gt",
        type=float,
        default=LIGHT_POSTPROCESS_FORCE_DEV_MAX_GT,
        help="Postprocess force deviation threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        summary = train_experiment(
            root=args.root,
            output_dir=args.output_dir,
            seed=args.seed,
            max_records=args.max_records,
            backend=args.backend,
            disagreement_mode=args.disagreement_mode,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "rank":
        ranked = rank_models_for_structure(
            bundle_path=args.bundle,
            structure_db=args.structure_db,
            structure_id=args.structure_id,
        )
        print(ranked.head(args.top_k).to_string(index=False))
        return

    if args.command == "rank-extxyz":
        summary = rank_extxyz_candidates(
            bundle_path=args.bundle,
            xyz_path=args.xyz,
            root=args.root,
            output_dir=args.output_dir,
            top_k=args.top_k,
            limit=args.limit,
            every_n=args.every_n,
            offset=args.offset,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "analyze":
        summary = run_policy_analysis(root=args.root, bundle_path=args.bundle, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "compare":
        summary = run_backend_comparison(
            root=args.root,
            output_dir=args.output_dir,
            seed=args.seed,
            max_records=args.max_records,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "plot":
        summary = generate_all_visualizations(root=args.root, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "tables":
        summary = generate_summary_tables(root=args.root, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "stability":
        summary = run_stability_analysis(
            root=args.root,
            output_dir=args.output_dir,
            backend=args.backend,
            seeds=args.seeds,
            policy_name=args.policy,
            n_bootstrap=args.n_bootstrap,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "route":
        thresholds = None
        if any(value is not None for value in (args.threshold_a, args.threshold_b, args.threshold_c)):
            thresholds = {
                "A": args.threshold_a if args.threshold_a is not None else 0.99,
                "B": args.threshold_b if args.threshold_b is not None else 0.99,
                "C": args.threshold_c if args.threshold_c is not None else 0.99,
            }
        decisions = route_structures(
            bundle_path=args.bundle,
            structure_db=args.structure_db,
            model_dbs=args.model_db,
            structure_ids=args.structure_id,
            thresholds=thresholds,
        )
        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            decisions.to_csv(args.output_csv, index=False)
        print(decisions.to_string(index=False))
        return

    if args.command == "ar-maxforce":
        summary = run_ar_maxforce_experiment(
            root=args.root,
            output_dir=args.output_dir,
            seed=args.seed,
            backend=args.backend,
            disagreement_mode=args.disagreement_mode,
            precision_target=args.precision_target,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "ar-maxforce-cv":
        summary = run_ar_maxforce_cross_validation(
            root=args.root,
            output_dir=args.output_dir,
            seed=args.seed,
            backend=args.backend,
            disagreement_mode=args.disagreement_mode,
            precision_target=args.precision_target,
            n_splits=args.n_splits,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "ar-route":
        decisions = route_ar_structures(
            bundle_path=args.bundle,
            structure_db=args.structure_db,
            model_dbs=args.model_db,
            structure_ids=args.structure_id,
            threshold=args.threshold,
            postprocess_light=args.postprocess_light,
            postprocess_confidence_lt=args.postprocess_confidence_lt,
            postprocess_force_dev_max_gt=args.postprocess_force_dev_max_gt,
        )
        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            decisions.to_csv(args.output_csv, index=False)
        print(decisions.to_string(index=False))
        return

    if args.command == "ar-postprocess":
        summary = run_ar_postprocess_analysis(
            root=args.root,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            confidence_lt=args.confidence_lt,
            force_dev_max_gt=args.force_dev_max_gt,
        )
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()
