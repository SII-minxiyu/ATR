#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat \
  /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/Mg/POTCAR  \
  /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/Si/POTCAR   > POTCAR
echo "Wrote POTCAR for POSCAR element order: Mg Si"
