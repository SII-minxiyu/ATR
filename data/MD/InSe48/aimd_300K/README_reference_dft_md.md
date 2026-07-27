# Reference DFT-MD Runs for InSe 48 Atoms

Created run directories:

- `run_001_smoke_300K_20steps`: NVT, 300 K, 20 steps, 1 fs
- `run_002_nvt_300K_1ps`: NVT, 300 K, 1000 steps, 1 fs
- `run_003_nvt_300K_5ps`: NVT, 300 K, 5000 steps, 1 fs template

## Resource Summary

- Current host: `inse--ccba6b3bed49-6jdszfak3p`
- Online CPUs visible in current environment: 64 from `nproc`; `lscpu` reports 128 configured with CPUs 0-63 online
- CPU model: Intel Xeon CPU Max 9462, 2 sockets x 32 cores, 1 thread/core online
- Memory: 503 GiB total; about 430 GiB available during setup
- No SLURM/PBS commands were found in the current shell
- oneAPI setup path works: `/inspire/hdd/global_user/luomingxiang-240108540155/intel/oneapi/setvars.sh`
- Intel MPI `mpirun` works after oneAPI is sourced
- VASP binary links successfully after oneAPI is sourced
- `mpirun -np 2 hostname` works outside the command sandbox with `I_MPI_FABRICS=shm`

## Recommended Parallel Settings

Use single-node pure MPI first:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm
NP=16 ./run.sh
```

Use `NP=16` for the smoke test. If stable, test `NP=32` on the same 20-step case before committing to 1000 steps. For this 48-atom Gamma-only MD, more than 32 ranks may under-scale unless the measured time per MD step proves otherwise.

In `INCAR`, use:

- `KPAR=1` because there is only one k-point
- `NCORE=4` for the initial 16/32-rank tests
- Do not set `NPAR`; VASP derives it from `NCORE`

Use `I_MPI_FABRICS=shm` in this detected single-node environment. Use `I_MPI_FABRICS=shm:ofi` only for a real multi-node job with verified OFI fabric and scheduler allocation.

## POTCAR Recommendation

The POSCAR element order is `In Se`, so the POTCAR must be in the same order. Recommended:

```bash
cat /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/In_d/POTCAR \
    /inspire/hdd/global_user/luomingxiang-240108540155/app/pot/Se/POTCAR > POTCAR
```

Use `In_d` because In 4d semicore states can affect bonding and forces; this is the safer choice for DFT-MD reference data. Use standard `Se` for the same PAW/PBE family. Avoid mixing GW/harder variants unless the whole workflow is deliberately benchmarked with those potentials.

The current setup account could see the `In`, `In_d`, and `Se` directories but could not read their contents, so POTCAR creation may require running as the account with pseudopotential access.

## Run Order

1. Generate `POTCAR` in `run_001_smoke_300K_20steps`.
2. Run `NP=16 ./run.sh`.
3. Check `vasp.out`, `OSZICAR`, `OUTCAR`, `XDATCAR`, and `CONTCAR`.
4. If the smoke test is stable, copy `CONTCAR` to `../run_002_nvt_300K_1ps/POSCAR`, generate the same POTCAR there, and run 1000 steps.

I can later inspect the completed VASP files for normal termination, SCF behavior, temperature fluctuations, energy drift, close contacts, and whether it is sensible to extend the trajectory.
