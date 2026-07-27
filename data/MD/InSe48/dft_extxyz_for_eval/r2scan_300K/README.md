# R2SCAN 300 K Dataset

Source trajectories:

- `../../reference_dft_md/run_005_r2scan_nvt_300K_1ps`
- `../../reference_dft_md/run_006_r2scan_nvt_300K_extend_1ps`

Files:

- `InSe48_r2scan_300K_part1_1ps_all.extxyz`
  - 1000 frames from `run_005`
- `InSe48_r2scan_300K_part2_1ps_all.extxyz`
  - 1000 frames from `run_006`
- `InSe48_r2scan_300K_2ps_all.extxyz`
  - combined 2000 frames from part1 + part2
  - 1 fs spacing
- `InSe48_r2scan_300K_2ps_stride20.extxyz`
  - 100 frames
  - every 20 fs
  - recommended compact 300 K test set
- `InSe48_r2scan_300K_2ps_stride200.extxyz`
  - 10 frames
  - every 200 fs
  - sparse benchmark-style sampling

Each extxyz frame contains cell, PBC, positions, energy, and forces parsed from VASP OUTCAR by ASE.
