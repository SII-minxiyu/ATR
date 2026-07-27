# R2SCAN 300 K 5 ps Dataset

Source trajectories:

- `../../reference_dft_md/run_002_r2scan_nvt_300K_2ps/OUTCAR`
- `../../reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps/OUTCAR`

Files:

- `HgInSn48_r2scan_300K_5ps_all.extxyz`
  - 5000 frames
  - full 5 ps trajectory
  - 1 fs spacing
- `HgInSn48_r2scan_300K_5ps_prod_after500fs_all.extxyz`
  - 4500 frames
  - first 0.5 ps removed
  - 1 fs spacing
- `HgInSn48_r2scan_300K_5ps_prod_after500fs_stride20.extxyz`
  - 225 frames
  - first 0.5 ps removed
  - every 20 fs
- `HgInSn48_r2scan_300K_5ps_prod_after500fs_stride200.extxyz`
  - 23 frames
  - first 0.5 ps removed
  - every 200 fs

Each extxyz frame contains cell, PBC, positions, energy, and forces parsed from VASP OUTCAR by ASE.

The production subset starts at source step 501, which corresponds to dropping the first 500 MD frames as thermalization.
