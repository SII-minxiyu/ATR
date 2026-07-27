# R2SCAN 300 K Dataset

Source trajectory:

- `../../reference_dft_md/run_002_r2scan_nvt_300K_2ps`

Files:

- `HgInSn48_r2scan_300K_2ps_all.extxyz`
  - 2000 frames
  - 1 fs spacing
- `HgInSn48_r2scan_300K_2ps_stride20.extxyz`
  - 100 frames
  - every 20 fs
  - recommended compact 300 K test set
- `HgInSn48_r2scan_300K_2ps_stride200.extxyz`
  - 10 frames
  - every 200 fs
  - sparse benchmark-style sampling

Each extxyz frame contains cell, PBC, positions, energy, and forces parsed from VASP OUTCAR by ASE.
