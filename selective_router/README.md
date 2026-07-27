# Server Deploy Bundle

This bundle contains the minimal code and trained bundles needed for large-scale teacher routing on new structures.

## Included models

- `models/candidate_ranker_top5.joblib`
  - Structure-only candidate ranking model.
  - Trained only on the final selected 5 strong teachers.
  - Use this first on raw `xyz/extxyz` to get top teacher candidates.
- `models/ar_router_seed2026.joblib`
  - Final practical `A/R` router bundle.
  - This corresponds to:
    - `A`: `|ΔE| < 0.1 eV/atom` and `force_max < 1.0 eV/A`
    - router backend: `XGBoost + top5 strong teachers`
  - Recommended deployment mode:
    - `top5` strong teachers only
    - lightweight postprocess enabled
    - `confidence < 0.45`
    - `router__target_force_dev_max > 0.15`

## Recommended strong teachers

Use these 5 teacher outputs for final routing whenever possible:

1. `7net-omni-i12-mp_r2scan`
2. `7net-omni-mp_r2scan`
3. `7net-omni-i12-matpes_r2scan`
4. `7net-omni-matpes_r2scan`
5. `mace-mh-1-matpes_r2scan`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_server.txt
```

## Workflow

### 1. Put your data into `data/`

Examples:

- `data/selected_train_90.xyz`
- `data/7net-omni-i12-mp_r2scan.extxyz`
- `data/7net-omni-i12-matpes_r2scan.extxyz`
- `data/7net-omni-mp_r2scan.extxyz`
- `data/mace-mh-1-matpes_r2scan.extxyz`
- `data/7net-omni-matpes_r2scan.extxyz`

### 2. Candidate ranking directly from xyz

```bash
python run_selective_router.py rank-extxyz \
  --bundle models/candidate_ranker_top5.joblib \
  --xyz data/selected_train_90.xyz \
  --root . \
  --output-dir outputs/rank_candidates \
  --top-k 5
```

Output:

- `outputs/rank_candidates/candidate_topk.csv`
- `outputs/rank_candidates/summary.json`

### 3. Convert structure xyz to ASE db

```bash
python tools/extxyz_to_ase_db.py \
  --xyz data/selected_train_90.xyz \
  --out-db data/selected_train_90.db \
  --overwrite
```

### 4. Convert each teacher prediction extxyz to ASE db

```bash
python tools/extxyz_to_ase_db.py --xyz data/7net-omni-i12-mp_r2scan.extxyz --out-db data/7net-omni-i12-mp_r2scan.db --overwrite
python tools/extxyz_to_ase_db.py --xyz data/7net-omni-i12-matpes_r2scan.extxyz --out-db data/7net-omni-i12-matpes_r2scan.db --overwrite
python tools/extxyz_to_ase_db.py --xyz data/7net-omni-mp_r2scan.extxyz --out-db data/7net-omni-mp_r2scan.db --overwrite
python tools/extxyz_to_ase_db.py --xyz data/mace-mh-1-matpes_r2scan.extxyz --out-db data/mace-mh-1-matpes_r2scan.db --overwrite
python tools/extxyz_to_ase_db.py --xyz data/7net-omni-matpes_r2scan.extxyz --out-db data/7net-omni-matpes_r2scan.db --overwrite
```

### 5. Final routing + reject

```bash
python run_selective_router.py ar-route \
  --bundle models/ar_router_seed2026.joblib \
  --structure-db data/selected_train_90.db \
  --model-db data/7net-omni-i12-mp_r2scan.db \
  --model-db data/7net-omni-mp_r2scan.db \
  --model-db data/7net-omni-i12-matpes_r2scan.db \
  --model-db data/7net-omni-matpes_r2scan.db \
  --model-db data/mace-mh-1-matpes_r2scan.db \
  --output-csv outputs/ar_route_decisions.csv \
  --postprocess-light \
  --postprocess-confidence-lt 0.45 \
  --postprocess-force-dev-max-gt 0.15
```

Output:

- `outputs/ar_route_decisions.csv`

Each row gives the final `accept/reject` decision, chosen teacher, and confidence.

### 5b. Direct routing from 5 teacher ExtXYZ files (no `.db` conversion)

If the five teacher prediction files are already aligned and correspond to the
same structure shard, you can skip `.db` conversion and route directly:

```bash
python tools/route_extxyz_top5.py \
  --bundle models/ar_router_seed2026.joblib \
  --teacher 7net-omni-i12-mp_r2scan=/path/to/7net-omni-i12-mp_r2scan.extxyz \
  --teacher 7net-omni-mp_r2scan=/path/to/7net-omni-mp_r2scan.extxyz \
  --teacher 7net-omni-i12-matpes_r2scan=/path/to/7net-omni-i12-matpes_r2scan.extxyz \
  --teacher 7net-omni-matpes_r2scan=/path/to/7net-omni-matpes_r2scan.extxyz \
  --teacher mace-mh-1-matpes_r2scan=/path/to/mace-mh-1-matpes_r2scan.extxyz \
  --output-csv outputs/ar_route_decisions.csv \
  --output-accepted-xyz outputs/ar_route_labeled_accept_only.extxyz \
  --postprocess-light \
  --postprocess-confidence-lt 0.45 \
  --postprocess-force-dev-max-gt 0.15
```

This writes:

- `outputs/ar_route_decisions.csv`
- `outputs/ar_route_decisions.summary.json`
- `outputs/ar_route_labeled_accept_only.extxyz`

The accepted extxyz contains the chosen teacher's predicted energy/forces, plus route labels in `atoms.info`.

### 6. Export routed structures back to ExtXYZ

You can use any one of the teacher `.db` files as `--structure-db`, because the
five teacher prediction databases correspond to the same structure set.

Export all structures with route labels:

```bash
python tools/export_route_decisions_to_extxyz.py \
  --structure-db data/7net-omni-i12-mp_r2scan.db \
  --decisions-csv outputs/ar_route_decisions.csv \
  --output-xyz outputs/ar_route_labeled_all.extxyz
```

Export only accepted structures:

```bash
python tools/export_route_decisions_to_extxyz.py \
  --structure-db data/7net-omni-i12-mp_r2scan.db \
  --decisions-csv outputs/ar_route_decisions.csv \
  --output-xyz outputs/ar_route_labeled_accept_only.extxyz \
  --accepted-only
```

The exported `extxyz` contains per-structure labels in `atoms.info`:

- `route_status`
- `route_chosen_model`
- `route_chosen_label`
- `route_confidence`
- `route_postprocess_reject`
- `route_postprocess_reason`

## Notes

- `rank-extxyz` works directly on `xyz/extxyz`.
- Final `ar-route` currently expects ASE `.db` files.
- The postprocess rule is optional, but it is the recommended practical deployment mode for cleaner high-purity selection.
