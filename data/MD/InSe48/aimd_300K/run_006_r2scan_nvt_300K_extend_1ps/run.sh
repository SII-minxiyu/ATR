#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

set +u
source /inspire/hdd/global_user/luomingxiang-240108540155/intel/oneapi/setvars.sh intel64
set -u

ulimit -s unlimited

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export KMP_STACKSIZE=256m
export KMP_AFFINITY=disabled
export I_MPI_FABRICS=shm
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core

NP="${NP:-16}"
VASP="/inspire/hdd/global_user/luomingxiang-240108540155/app/vasp.6.5.0/bin/vasp_std"

if [[ ! -f POTCAR ]]; then
  echo "ERROR: POTCAR is missing." >&2
  exit 2
fi

rm -f vasp.pid

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "NP=${NP}"
echo "I_MPI_FABRICS=${I_MPI_FABRICS}"
echo "VASP=${VASP}"

setsid bash -c 'echo $$ > vasp.pid; exec mpirun -np "$1" "$2" > vasp.out 2>&1' _ "${NP}" "${VASP}" &
for _ in {1..20}; do
  [[ -s vasp.pid ]] && break
  sleep 0.1
done
echo "Started VASP PID $(cat vasp.pid). Monitor with: tail -f vasp.out OSZICAR"
