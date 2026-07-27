#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat \
  /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/Hg/POTCAR \
  /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/In_d/POTCAR \
  /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/Sn_d/POTCAR   > POTCAR
echo "Wrote POTCAR for POSCAR element order: Hg In_d Sn_d"
