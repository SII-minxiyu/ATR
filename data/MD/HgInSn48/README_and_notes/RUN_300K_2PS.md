# HgInSn 48-Atom 300 K r2SCAN MD Run Guide

This folder contains the HgInSn 48-atom VASP r2SCAN DFT-MD setup.

Main contents:
- `structures/`: selected initial extxyz structures.
- `inputs/POSCAR`: base VASP structure.
- `reference_dft_md/run_001_r2scan_smoke_300K_20steps`: completed 20-step smoke test.
- `reference_dft_md/run_002_r2scan_nvt_300K_2ps`: completed 300 K, 2 ps production test.
- `reference_dft_md/run_003_r2scan_nvt_450K_2ps`: prepared for later 450 K work, do not run yet unless needed.
- `reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps`: ready-to-run 300 K continuation from 2 ps to 5 ps total.

Current target:
- Run `run_004_r2scan_nvt_300K_extend_3ps`.
- Settings: r2SCAN, Gamma-only, NVT, 300 K, 3000 steps, 1 fs timestep.
- Initial structure is copied from `run_002_r2scan_nvt_300K_2ps/CONTCAR`.
- `POTCAR` is already present in the run directory.
- POTCAR order is `Hg In_d Sn_d`.

Run 3 ps continuation on current 20-core environment:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps
chmod +x run.sh
NP=16 ./run.sh
```

Run 3 ps continuation on 55-core / 300 GB server:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps
chmod +x run.sh
./run.sh
```

Monitor:

```bash
tail -f OSZICAR vasp.out
```

Quick status check:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_002_r2scan_nvt_300K_2ps
tail -n 20 OSZICAR
tail -n 40 vasp.out
```

Expected cost from smoke test:
- `NP=16`: about 55 s/step average in the 20-step smoke test.
- 2000 steps: about 31 hours on a similar 16-rank run, depending on machine load.
- For the new 3000-step continuation, expect roughly 46 hours at 55 s/step with `NP=16`; on the 48-rank server it should be faster, but confirm from the first 20-50 steps.
