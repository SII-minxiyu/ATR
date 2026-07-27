#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar}"
LOCAL_CHGNET="${LOCAL_CHGNET:-/inspire/hdd/global_user/luomingxiang-240108540155/chgnet}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-10000}"
LOGINTERVAL="${LOGINTERVAL:-10}"

export PYTHONPATH="${LOCAL_CHGNET}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"${PYTHON}" scripts/run_chgnet_mlmd.py \
  --initial starts/MgSi44_frame_1001.extxyz \
  --checkpoint "${CHECKPOINT}" \
  --out-dir runs/run_001_frame1001_300K_10ps \
  --steps "${STEPS}" \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval "${LOGINTERVAL}" \
  --device "${DEVICE}" \
  --thermostat Berendsen
