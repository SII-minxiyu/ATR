#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar}"
LOCAL_CHGNET="${LOCAL_CHGNET:-/inspire/hdd/global_user/luomingxiang-240108540155/chgnet}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-10000}"
LOGINTERVAL="${LOGINTERVAL:-10}"
SEED="${SEED:-20260514}"

export PYTHONPATH="${LOCAL_CHGNET}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"${PYTHON}" scripts/run_chgnet_mlmd.py \
  --initial starts/HgInSn48_source_step_1047_1p046ps_mlmd_base_preferred_official_watch.extxyz \
  --checkpoint "${CHECKPOINT}" \
  --out-dir runs/run_005_step1047_base_300K_10ps \
  --steps "${STEPS}" \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval "${LOGINTERVAL}" \
  --device "${DEVICE}" \
  --thermostat Berendsen \
  --seed "${SEED}"

"${PYTHON}" scripts/plot_mlmd_diagnostics.py \
  --run-dir runs/run_005_step1047_base_300K_10ps
