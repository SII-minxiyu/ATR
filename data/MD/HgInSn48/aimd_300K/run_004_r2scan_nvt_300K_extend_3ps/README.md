# R2SCAN 300 K Continuation, 3 ps

This run extends the existing HgInSn 300 K R2SCAN trajectory from 2 ps to 5 ps total.

- Previous trajectory:
  `../run_002_r2scan_nvt_300K_2ps`
- Initial structure:
  copied from `../run_002_r2scan_nvt_300K_2ps/CONTCAR`
- Functional: R2SCAN
- Ensemble: NVT
- Temperature: 300 K
- Steps: 3000
- Timestep: 1 fs
- Added time: 3 ps
- Total 300 K R2SCAN time after completion: 5 ps
- Default MPI ranks: 48
- K-points: Gamma-only
- POTCAR order: Hg, In_d, Sn_d

Run on the 55-core / 300 GB server:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_004_r2scan_nvt_300K_extend_3ps
chmod +x run.sh
./run.sh
```

Monitor:

```bash
tail -f OSZICAR vasp.out
```

Quick progress check:

```bash
grep -E '^[[:space:]]*[0-9]+[[:space:]]+T=' OSZICAR | tail
```

To use fewer ranks for a local test:

```bash
NP=16 ./run.sh
```
