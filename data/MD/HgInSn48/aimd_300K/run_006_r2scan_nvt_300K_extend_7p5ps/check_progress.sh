#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Directory: $(pwd)"
echo "Host: $(hostname)"
echo "Date: $(date -Is)"

if [[ -f OSZICAR ]]; then
  steps=$(grep -Ec '^[[:space:]]*[0-9]+[[:space:]]+T=' OSZICAR || true)
  echo "MD steps in OSZICAR: ${steps}/3000"
  echo "Last MD lines:"
  grep -E '^[[:space:]]*[0-9]+[[:space:]]+T=' OSZICAR | tail -5 || true
else
  echo "OSZICAR not found yet."
fi

if [[ -f OUTCAR ]]; then
  if grep -q 'General timing and accounting informations' OUTCAR; then
    echo "OUTCAR status: finished"
  else
    echo "OUTCAR status: running or incomplete"
  fi
else
  echo "OUTCAR not found yet."
fi

echo "Running VASP/MPI processes visible here:"
pgrep -af 'vasp_std|mpirun|mpiexec|hydra' || true
