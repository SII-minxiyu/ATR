#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar}"
LOCAL_CHGNET="${LOCAL_CHGNET:-/inspire/hdd/global_user/luomingxiang-240108540155/chgnet}"

export PYTHONPATH="${LOCAL_CHGNET}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - <<PY
from pathlib import Path
print("python ok")
for mod in ("torch", "ase", "pymatgen", "chgnet"):
    try:
        m = __import__(mod)
        print(f"{mod}: OK {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"{mod}: FAIL {exc!r}")
ckpt = Path("${CHECKPOINT}")
print(f"checkpoint exists: {ckpt.is_file()} {ckpt}")
print("PYTHONPATH includes local CHGNet:", "${LOCAL_CHGNET}")
PY
