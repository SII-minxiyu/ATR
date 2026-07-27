# InSe 48-Atom MD Workspace

Purpose: collect files for 48-atom InSe molecular dynamics tests and later comparison against a DFT-MD reference trajectory.

## Layout

- `structures/`: source structures, currently `InSe_48_initial.extxyz`.
- `inputs/`: VASP-ready inputs, currently `POSCAR`.
- `reference_dft_md/`: planned DFT/R2SCAN MD runs.
- `model_md/`: model-driven MD runs.
- `analysis/`: RDF, MSD, energy drift, temperature, and trajectory comparison outputs.
- `logs/`: run logs and environment notes.

## Current Structure

- Formula: `In24Se24`
- Atoms: 48
- Cell lengths: `16.12477262 16.12477262 8.06238631 Angstrom`
- Source: `test_data/selected_initial_InSe_lowforce_2x2x1.extxyz`

## Environment Check

- VASP binaries found:
  - `/inspire/hdd/global_user/luomingxiang-240108540155/app/vasp.6.5.0/bin/vasp_std`
  - `/inspire/hdd/global_user/luomingxiang-240108540155/app/vasp.6.5.0/bin/vasp_gam`
  - `/inspire/hdd/global_user/luomingxiang-240108540155/app/vasp.6.5.0/bin/vasp_ncl`
- `vaspkit` found:
  - `/inspire/hdd/global_user/luomingxiang-240108540155/app/vaspkit.1.5.1/bin/vaspkit`
- Direct VASP execution currently fails because Intel runtime libraries are missing from the active environment, specifically `libimf.so`.
- ASE is available from `/opt/conda/envs/test/bin/ase`.
- Python packages available: `ase`, `pymatgen`, `chgnet`, `mace`, `torch`.
- Not found in active PATH: `mpirun`, `srun`, `sbatch`, `qsub`.
