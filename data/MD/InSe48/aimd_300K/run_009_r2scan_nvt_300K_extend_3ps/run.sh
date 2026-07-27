#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ONEAPI="${ONEAPI:-/inspire/hdd/global_user/luomingxiang-240108540155/intel/oneapi/setvars.sh}"
VASP="${VASP:-/inspire/hdd/global_user/luomingxiang-240108540155/app/vasp.6.5.0/bin/vasp_std}"

set +u
if [[ -z "${I_MPI_ROOT:-}" || -z "${MKLROOT:-}" ]]; then
  source "${ONEAPI}" intel64
else
  echo "oneAPI environment already active: I_MPI_ROOT=${I_MPI_ROOT:-}, MKLROOT=${MKLROOT:-}"
fi
set -u

ulimit -s unlimited

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export KMP_STACKSIZE=256m
export KMP_AFFINITY=disabled
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core

# Production setting for the 55-core / 300 GB server.
# Use 48 MPI ranks with NCORE=4 for this 48-atom Gamma-only R2SCAN MD.
NP="${NP:-48}"

if [[ ! -f POSCAR || ! -f POTCAR || ! -f INCAR || ! -f KPOINTS ]]; then
  echo "ERROR: POSCAR/POTCAR/INCAR/KPOINTS must all exist in $(pwd)" >&2
  exit 2
fi

if pgrep -af 'vasp_std|mpirun|mpiexec|hydra' | grep -v pgrep >/dev/null; then
  echo "ERROR: Existing VASP/MPI process found. Check before starting another job." >&2
  pgrep -af 'vasp_std|mpirun|mpiexec|hydra' >&2 || true
  exit 3
fi

rm -f vasp.pid
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "nproc: $(nproc)"
echo "NP=${NP}"
echo "I_MPI_FABRICS=${I_MPI_FABRICS}"
echo "VASP=${VASP}"

setsid bash -c 'echo $$ > vasp.pid; exec mpirun -np "$1" "$2" > vasp.out 2>&1' _ "${NP}" "${VASP}" &
for _ in {1..20}; do
  [[ -s vasp.pid ]] && break
  sleep 0.1
done
echo "Started VASP launcher PID $(cat vasp.pid). Monitor with: tail -f OSZICAR vasp.out"
