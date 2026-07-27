# MgSi44 R2SCAN MD Test Dataset

This directory stores extxyz exports for ML potential testing.

Source VASP run:

`../reference_dft_md/run_002_r2scan_nvt_300K_2ps/OUTCAR`

## 300 K

Directory:

`r2scan_300K/`

Files:

- `MgSi44_r2scan_300K_2ps_all.extxyz`
  - 2000 frames
  - every MD step
  - 1 fs spacing
- `MgSi44_r2scan_300K_2ps_stride20.extxyz`
  - 100 frames
  - every 20 fs
  - recommended compact 300 K test set
- `MgSi44_r2scan_300K_2ps_stride200.extxyz`
  - 10 frames
  - every 200 fs
  - sparse benchmark-style sampling

Each extxyz frame contains cell, PBC, positions, energy, and forces parsed from VASP OUTCAR by ASE.
