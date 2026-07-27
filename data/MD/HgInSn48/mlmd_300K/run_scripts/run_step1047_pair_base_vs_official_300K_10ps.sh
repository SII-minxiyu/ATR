#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash run_step1047_base_300K_10ps.sh
bash run_step1047_official_300K_10ps.sh
