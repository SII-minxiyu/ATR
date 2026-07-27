# InSe48 R2SCAN MD Test Dataset

This directory stores extxyz exports for ML potential testing.

Source VASP runs are under:

`../reference_dft_md`

## 600 K

Directory:

`r2scan_600K/`

Source:

`../reference_dft_md/run_008_r2scan_nvt_600K_2ps/OUTCAR`

Files:

- `InSe48_r2scan_600K_2ps_all.extxyz`
  - 2000 frames
  - every MD step
  - 1 fs spacing
- `InSe48_r2scan_600K_2ps_stride20.extxyz`
  - 100 frames
  - every 20 fs
  - useful as a compact test set
- `InSe48_r2scan_600K_2ps_stride200.extxyz`
  - 10 frames
  - every 200 fs
  - similar sparse sampling interval to some published AIMD benchmarks

Each extxyz frame contains cell, PBC, positions, energy, and forces parsed from VASP OUTCAR by ASE.

## Re-export

Use:

```bash
../scripts/export_vasp_md_to_extxyz.py \
  --input ../reference_dft_md/run_008_r2scan_nvt_600K_2ps/OUTCAR \
  --output r2scan_600K/InSe48_r2scan_600K_2ps_stride20.extxyz \
  --stride 20 \
  --temperature 600 \
  --source-run run_008_r2scan_nvt_600K_2ps \
  --metadata r2scan_600K/metadata_stride20.json
```

For model reporting, keep 300 K and 600 K errors separate, then optionally report a combined score.

## 300 K

Directory:

`r2scan_300K/`

Source:

- `../reference_dft_md/run_005_r2scan_nvt_300K_1ps/OUTCAR`
- `../reference_dft_md/run_006_r2scan_nvt_300K_extend_1ps/OUTCAR`

Main files:

- `InSe48_r2scan_300K_2ps_all.extxyz`
  - 2000 frames
  - 1 fs spacing
- `InSe48_r2scan_300K_2ps_stride20.extxyz`
  - 100 frames
  - every 20 fs
- `InSe48_r2scan_300K_2ps_stride200.extxyz`
  - 10 frames
  - every 200 fs
