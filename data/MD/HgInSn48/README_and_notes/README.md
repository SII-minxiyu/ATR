# HgInSn 48-Atom MD Workspace

Purpose: collect files for short r2SCAN DFT-MD reference tests and later model-MD comparison.

## Why This System

260w clean accepted has 2703 Hg-In-Sn frames while MP-r2SCAN has 0; useful harder case, but heavier/softer and less interpretable than Ag-Al-Zn.

## Layout

- `structures/`: selected low-force source frame and repeated initial supercell.
- `inputs/`: VASP-ready `POSCAR` for the repeated supercell.
- `reference_dft_md/`: prepared r2SCAN DFT-MD run folders.
- `model_md/`: model-driven MD runs.
- `analysis/`: RDF, MSD, energy drift, force/energy error, and trajectory comparison outputs.
- `logs/`: run logs and environment notes.

## Current Structure

- Source formula/counts: `{'In': 5, 'Sn': 3, 'Hg': 4}`
- Source atoms: `12`
- Repeat: `2 x 2 x 1`
- Supercell atoms: `48`
- Initial supercell: `structures/HgInSn_48_initial.extxyz`
- VASP POSCAR: `inputs/POSCAR`
- Selected-frame max force: `0.131426 eV/A`
- Selected-frame energy: `-383.6636962890625` eV
- Selection note: representative most-common Hg-In-Sn formula in 260w clean accepted (`Hg4In5Sn3`, 558 frames); not the absolute-lowest-force In-rich formula.
- Source comment: `Lattice="8.25354533 -0.76934447 0.00011423 -0.8316411 6.80734777 -3.03510777 5.651e-05 -0.00011408 6.07009031" Properties=species:S:1:pos:R:3:forces:R:3 id=agm005695580-r2scan-198 immutable_id=agm005695580 energy=-383.6636962890625 pbc="T T T"`

## Temperature Recommendation

Use 300 K as the main first test because the current training data may not be rich in high-energy/high-force configurations. Use 450 K as a moderate challenge only after 300 K is stable. Treat 600 K as a pressure test, not as the first-pass accuracy conclusion.

Recommended order:

1. `run_001_r2scan_smoke_300K_20steps`
2. `run_002_r2scan_nvt_300K_2ps`
3. `run_003_r2scan_nvt_450K_2ps`

For the 300 K reference trajectory, `run_002` has been extended by `run_004_r2scan_nvt_300K_extend_3ps` to make a total 5 ps trajectory.

## POTCAR

POSCAR is written with sorted element order. Use POTCAR order:

`Hg In_d Sn_d`

If direct `cat` fails because of permissions, configure `~/.vaspkit` `PBE_PATH` to `/inspire/hdd/global_user/luomingxiang-240108540155/app/pot` and use vaspkit `01 -> 103` inside the run directory.
