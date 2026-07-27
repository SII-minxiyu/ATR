python run_selective_router.py rank-extxyz \
  --bundle models/candidate_ranker_top5.joblib \
  --xyz data/selected_train_90.xyz \
  --root . \
  --output-dir outputs/rank_candidates \
  --top-k 5

python tools/extxyz_to_ase_db.py \
  --xyz data/selected_train_90.xyz \
  --out-db data/selected_train_90.db \
  --overwrite

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

python tools/export_route_decisions_to_extxyz.py \
  --structure-db data/7net-omni-i12-mp_r2scan.db \
  --decisions-csv outputs/ar_route_decisions.csv \
  --output-xyz outputs/ar_route_labeled_accept_only.extxyz \
  --accepted-only
