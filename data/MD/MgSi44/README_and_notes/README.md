# MgSi 44-Atom MD Workspace

Purpose: short r2SCAN DFT-MD reference tests and later model-MD comparison for `Mg-Si`.

## Why This System

MP-r2SCAN, 260w accepted, and LeMat18k all contain this system. It is the most practical common candidate: light elements, cheaper r2SCAN MD, and good for checking binary alloy/semiconductor local environments.

## What MD Can Tell Us

Short NVT MD can test finite-temperature stability, Mg/Si local coordination changes, force smoothness, energy drift, RDF peak positions, and whether model errors grow under thermal distortion.

## Layout

- `structures/`: selected source structure and repeated initial supercell.
- `inputs/`: VASP-ready `POSCAR` for the repeated supercell.
- `reference_dft_md/`: prepared r2SCAN DFT-MD run folders.
- `model_md/`: model-driven MD runs.
- `analysis/`: RDF, MSD, energy drift, force/energy error, and trajectory comparison outputs.
- `logs/`: run logs and environment notes.

## Current Structure

- Source file: `/inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/server_deploy_router_bundle/coverage_projection_cached_pbe1500_acc150k/common_mp_260w_lemat_selected_systems/accepted260w_full_Mg-Si.extxyz`
- Selected source index in extracted file: `0`
- Source formula: `Mg12Si10`
- Source atoms: `22`
- Repeat: `2 x 1 x 1`
- Supercell atoms: `44`
- Initial supercell: `structures/MgSi_44_initial.extxyz`
- VASP POSCAR: `inputs/POSCAR`
- Available source labels: structure only; no energy/force arrays in this accepted extract.

## Temperature Recommendation

Use 300 K as the main first test. Use 450 K as a moderate challenge only after 300 K is stable. Treat 600 K, if added later, as a pressure test rather than the first-pass accuracy conclusion.

Recommended order:

1. `run_001_r2scan_smoke_300K_20steps`
2. `run_002_r2scan_nvt_300K_2ps`
3. `run_003_r2scan_nvt_450K_2ps`

## POTCAR

POSCAR is written with element order:

`Mg Si`

Use `./make_potcar.sh` inside each run directory, or use vaspkit `01 -> 103` with `PBE_PATH=/inspire/hdd/global_user/luomingxiang-240108540155/app/pot`.
