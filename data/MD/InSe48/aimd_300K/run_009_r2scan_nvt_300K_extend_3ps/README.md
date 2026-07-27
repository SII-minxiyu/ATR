# R2SCAN 300 K Continuation, 3 ps

This run extends the existing 300 K R2SCAN trajectory from 2 ps to 5 ps total.

- Previous trajectory:
  `../run_005_r2scan_nvt_300K_1ps` + `../run_006_r2scan_nvt_300K_extend_1ps`
- Initial structure:
  copied from `../run_006_r2scan_nvt_300K_extend_1ps/CONTCAR`
- Functional: R2SCAN
- Ensemble: NVT
- Temperature: 300 K
- Steps: 3000
- Timestep: 1 fs
- Added time: 3 ps
- Total 300 K R2SCAN time after completion: 5 ps
- Default MPI ranks: 48
- K-points: Gamma-only
- POTCAR order: In_d, Se

Run on the 55-core / 300 GB server:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_InSe_48atom_vasp/reference_dft_md/run_009_r2scan_nvt_300K_extend_3ps
./run.sh
```

Monitor:

```bash
tail -f OSZICAR vasp.out
```

Expected wall time is roughly 16-20 hours with 48 MPI ranks.
